import os
import time
import mujoco
import copy
import glfw
import pathlib
import cv2
import numpy as np
from threading import Lock

MUJOCO_VERSION = tuple(map(int,mujoco.__version__.split('.')))

from .transforms import (
    t2p,
    t2r,
    pr2t,
    r2quat,
    quat2r,
    r2w,
    rpy2r,
    meters2xyz,
    get_rotation_matrix_from_two_points,
)
from .utils import (
    trim_scale,
    compute_view_params,
    get_idxs,
    get_colors,
    get_monitor_size,
    TicTocClass,
)

class MinimalCallbacks:
    def __init__(self, hide_menus):
        self._gui_lock                   = Lock()
        self._button_left_pressed        = False
        self._button_right_pressed       = False
        self._left_double_click_pressed  = False
        self._right_double_click_pressed = False
        self._last_left_click_time       = None
        self._last_right_click_time      = None
        self._last_mouse_x               = 0
        self._last_mouse_y               = 0
        self._paused                     = False
        self._render_every_frame         = True
        self._time_per_render            = 1/60.0
        self._run_speed                  = 1.0
        self._loop_count                 = 0
        self._advance_by_one_step        = False
        # Keyboard 
        self._key_pressed                = None
        self._is_key_pressed             = False
        # Keyboard buffer
        self._key_pressed_set            = set()
        self._key_repeated_set           = set()
        
    def _key_callback(self, window, key, scancode, action, mods):
        """
            Key callback        
        """

        # Flags for key pressed 
        is_key_pressed  = (action==glfw.PRESS)
        is_key_released = (action==glfw.RELEASE)
        is_key_repeated = (action==glfw.REPEAT)
        
        # Add and discard keys
        if is_key_pressed:
            self._key_pressed_set.add(key)
        if is_key_repeated:
            self._key_repeated_set.add(key)
        if is_key_released:
            # Remove from pressed and repeated lists (if present)
            self._key_pressed_set.discard(key)
            self._key_repeated_set.discard(key)
        
        # Pause / resume handling (space)
        # if is_key_pressed and (key==glfw.KEY_SPACE) and (self._paused is not None):
        #     self._paused = not self._paused

        # Quit (escape)
        if (key==glfw.KEY_ESCAPE):
            glfw.set_window_should_close(self.window, True)

        # Store key pressed (legacy)
        self._key_pressed    = key 
        self._is_key_pressed = True
        
        # Return
        return

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert)
            else:
                mujoco.mjv_moveCamera(
                    self.model,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.cam)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # detect a left- or right- doubleclick
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()

            time_diff = (time_now - self._last_left_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()

            time_diff = (time_now - self._last_right_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        # set perturbation
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            # right: translate, left: rotate
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # perturbation onste: reset reference
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(
                    self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb
        # 3D release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)

class MuJoCoMinimalViewer(MinimalCallbacks):
    def __init__(
            self,
            model,
            data,
            mode              = 'window',
            title             = "MuJoCo Minimal Viewer",
            width             = None,
            height            = None,
            hide_menus        = True,
            maxgeom           = 10000,
            n_fig             = 1,
            perturbation      = True,
            use_rgb_overlay   = True,
            loc_rgb_overlay   = 'top right',
        ):
        super().__init__(hide_menus)

        self.model = model
        self.data = data
        self.render_mode = mode
        if self.render_mode not in ['window']:
            raise NotImplementedError(
                "Invalid mode. Only 'window' is supported.")

        # keep true while running
        self.is_alive = True

        self.CONFIG_PATH = pathlib.Path.joinpath(
            pathlib.Path.home(), ".config/mujoco_viewer/config.yaml")

        # glfw init
        glfw.init()

        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size

        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
            
        if self.render_mode == 'offscreen':
            glfw.window_hint(glfw.VISIBLE, 0)

        # Create window
        self.maxgeom = maxgeom
        self.window = glfw.create_window(
            width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
            self.window)

        # install callbacks only for 'window' mode
        if self.render_mode == 'window':
            window_width, _ = glfw.get_window_size(self.window)
            self._scale = framebuffer_width * 1.0 / window_width

            # set callbacks
            glfw.set_cursor_pos_callback(
                self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(
                self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam  = mujoco.MjvCamera()
        self.scn  = mujoco.MjvScene(self.model, maxgeom=self.maxgeom)
        self.pert = mujoco.MjvPerturb()

        self.ctx = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        width, height = glfw.get_framebuffer_size(self.window)
        
        # figures
        self.n_fig = n_fig
        self.figs  = []
        for idx in range(self.n_fig):
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.flg_extend = 1
            fig.figurergba = (1,1,1,0)
            fig.panergba   = (1,1,1,0.2)
            self.figs.append(fig)

        # get viewport
        self.viewport = mujoco.MjrRect(
            0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers = []
        
        # rgb image to overlay (legacy)
        self.use_rgb_overlay = use_rgb_overlay
        self.loc_rgb_overlay = loc_rgb_overlay

        # rgb images to overlay
        self.rgb_overlay_top_right    = None
        self.rgb_overlay_top_left     = None
        self.rgb_overlay_bottom_right = None
        self.rgb_overlay_bottom_left  = None        
        
        # Perturbation
        self.perturbation = perturbation

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                'Ran out of geoms. maxgeom: %d' %
                self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        # g.matid = -1 # newly added (by Jihwan, 2025-02-27)
        """
            mujoco version 3.2 is NOT backward-compatible
        """
        if MUJOCO_VERSION[1] == 1:
            """
                Following lines make error for mujoco version 3.2
            """
            g.texid        = -1
            g.texuniform   = 0
            g.texrepeat[0] = 1
            g.texrepeat[1] = 1
        
        g.emission    = 0
        g.specular    = 0.5
        g.shininess   = 0.5
        g.reflectance = 0
        g.type        = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:]     = np.ones(3) * 0.1
        g.mat[:]      = np.eye(3)
        g.rgba[:]     = np.ones(4)

        for key, value in marker.items():
            # setattr(g, key, value)
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                # assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)
            
        # Increment number of geoms
        self.scn.ngeom += 1
        return

    def apply_perturbations(self):
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def read_pixels(self, camid=None, depth=False):
        if self.render_mode == 'window':
            raise NotImplementedError(
                "Use 'render()' in 'window' mode.")

        if camid is not None:
            if camid == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camid

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
            self.window)
        # update scene
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn)
        # render
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        shape = glfw.get_framebuffer_size(self.window)

        if depth:
            rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb_img, depth_img, self.viewport, self.ctx)
            return (np.flipud(rgb_img), np.flipud(depth_img))
        else:
            img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            return np.flipud(img)

    def add_overlay(
            self,
            loc     = 'bottom left',
            gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT,
            text1   = '',
            text2   = '',
        ):
        """
            Add overlay
            loc: ['top','top right','top left','bottom','bottom right','bottom left']
            Usage:
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOPLEFT,text1='TopLeft')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOP,text1='Top')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_TOPRIGHT,text1='TopRight')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,text1='BottomLeft')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOM,text1='Bottom')
                env.viewer.add_overlay(gridpos=mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,text1='BottomRight')
        """
        if loc is not None:
            if loc == 'top': gridpos = mujoco.mjtGridPos.mjGRID_TOP
            elif loc == 'top right': gridpos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
            elif loc == 'top left': gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT
            elif loc == 'bottom': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOM
            elif loc == 'bottom right': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
            elif loc == 'bottom left': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
            
        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1
            self._overlay[gridpos][1] += text2    
        else:
            self._overlay[gridpos][0] += "\n" + text1
            self._overlay[gridpos][1] += "\n" + text2    
        # self._overlay[gridpos][0] += text1 + "\n"
        # self._overlay[gridpos][1] += text2 + "\n"
        
    def _create_overlay(self):
        """ 
            Overlay items
        """
        topleft     = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright    = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft  = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
        
        # self.add_overlay(
        #     gridpos = topleft,
        #     text1   = "A",
        #     text2   = "B",
        # )
    
    def add_line(
            self,
            fig_idx    = 0,
            line_idx   = 0,
            xdata      = np.linspace(0,1,mujoco.mjMAXLINEPNT),
            ydata      = np.zeros(mujoco.mjMAXLINEPNT),
            linergb    = (0,0,1),
            linename   = 'Line Name',
            figurergba = (1,1,1,0),
            panergba   = (1,1,1,0.2),
        ):
        """ 
            Add line to the internal figure
            Usage:
                xdata = np.linspace(start=0.0,stop=10.0,num=100)
                ydata = np.sin(xdata)
                env.viewer.add_line(
                    fig_idx=0,line_idx=0,xdata=xdata,ydata=ydata,linergb=(1,0,0),linename='Line 1')
                xdata = np.linspace(start=0.0,stop=10.0,num=100)
                ydata = np.cos(xdata)
                env.viewer.add_line(
                    fig_idx=0,line_idx=1,xdata=xdata,ydata=ydata,linergb=(0,0,1),linename='Line 2')
        """
        fig = self.figs[fig_idx]
        fig.figurergba  = figurergba
        fig.panergba    = panergba
        L = len(xdata) # this cannot exceed 'mujoco.mjMAXLINEPNT'
        for i in range(L):
            fig.linedata[line_idx][2*i]   = xdata[i]
            fig.linedata[line_idx][2*i+1] = ydata[i]
        fig.linergb[line_idx]  = linergb
        fig.linename[line_idx] = linename
        fig.linepnt[line_idx]  = L
        
    def add_rgb_overlay(self,rgb_img_raw,fix_ratio=False):
        """
            Set RGB image to render 
        """
        width,height = glfw.get_framebuffer_size(self.window)
        rgb_h,rgb_w = height//4,width//4
        self.rgb_overlay = np.zeros((rgb_h,rgb_w,3))
        (h,w) = self.rgb_overlay.shape[:2]
        if fix_ratio: # fix aspect ratio
            h_raw, w_raw = rgb_img_raw.shape[:2]
            # Calculate scale to preserve aspect ratio
            scale = min(w / w_raw, h / h_raw)
            new_w = int(w_raw * scale)
            new_h = int(h_raw * scale)
            # Resize the image while preserving the aspect ratio
            resized_img = cv2.resize(rgb_img_raw, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Create a black canvas with the target size
            padded_img = np.zeros((h, w, 3), dtype=np.uint8)
            # Calculate the top-left corner for centering the resized image
            x_offset = (w - new_w) // 2
            y_offset = (h - new_h) // 2
            # Place the resized image onto the canvas
            padded_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img
            rgb_img_rsz = padded_img  # Final resized and padded image
        else:
            rgb_img_rsz = cv2.resize(rgb_img_raw,(w,h),interpolation=cv2.INTER_NEAREST)
        self.rgb_overlay = rgb_img_rsz

    def plot_rgb_overlay(self,rgb=None,loc='top right'):
        """
            loc:['top right','top left','bottom right','bottom left']
        """
        w_window,h_window = glfw.get_framebuffer_size(self.window)
        h_overlay,w_overlay = h_window//4,w_window//4
        rgb_overlay = np.zeros((h_overlay,w_overlay,3))
        # Fix aspect ratio
        h_raw,w_raw = rgb.shape[:2]
        # Calculate scale to preserve aspect ratio
        scale = min(w_overlay/w_raw,h_overlay/h_raw)
        w_new = int(w_raw*scale)
        h_new = int(h_raw*scale)
        # Resize
        rgb_resized = cv2.resize(rgb,(w_new,h_new),interpolation=cv2.INTER_NEAREST)
        # Create a black canvas with the target size
        rgb_padded = np.zeros((h_overlay,w_overlay,3),dtype=np.uint8)
        # Calculate the top-left corner for centering the resized image
        x_offset = (w_overlay-w_new) // 2
        y_offset = (h_overlay-h_new) // 2
        # Place the resized image onto the canvas
        rgb_padded[y_offset:y_offset+h_new,x_offset:x_offset+w_new] = rgb_resized
        # Store the RGB overlay
        if loc=='top right':
            self.rgb_overlay_top_right = rgb_padded
        elif loc=='top left':
            self.rgb_overlay_top_left = rgb_padded
        elif loc=='bottom right':
            self.rgb_overlay_bottom_right = rgb_padded
        elif loc=='bottom left':
            self.rgb_overlay_bottom_left = rgb_padded
        else:
            print ("Invalid location for RGB overlay. Use 'top right', 'top left', 'bottom right', or 'bottom left'.")

    def reset_rgb_overlay(self,loc=None):
        """
            loc:['top right','top left','bottom right','bottom left']
        """
        if loc is None:
            self.rgb_overlay_top_right    = None
            self.rgb_overlay_top_left     = None
            self.rgb_overlay_bottom_right = None
            self.rgb_overlay_bottom_left  = None
        else:
            if loc=='top_right':
                self.rgb_overlay_top_right = None
            if loc=='top left':
                self.rgb_overlay_top_left = None
            if loc=='bottom right':
                self.rgb_overlay_bottom_right = None
            if loc=='bottom left':
                self.rgb_overlay_bottom_left = None
    
    def render(self):
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            
            # Fill overlay items
            self._create_overlay()
            
            # Render start
            render_start = time.time()
            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            with self._gui_lock:
                # update scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn)
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)
                
                # overlay items
                for gridpos, [t1, t2] in self._overlay.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        t1,
                        t2,
                        self.ctx)
                    
                # handle figures
                for idx,fig in enumerate(self.figs):
                    width_adjustment = width % 4
                    x = int(3 * width / 4) + width_adjustment
                    y = idx * int(height / 4)
                    viewport = mujoco.MjrRect(
                        x, y, int(width / 4), int(height / 4))
                    # Plot
                    mujoco.mjr_figure(viewport, fig, self.ctx)
                    
                # roverlay rgb images (legacy)
                if self.use_rgb_overlay:
                    rgb_h,rgb_w = height//4,width//4
                    if self.loc_rgb_overlay == 'top right':
                        left   = 3*rgb_w
                        bottom = 3*rgb_h
                    elif self.loc_rgb_overlay == 'top left':
                        left   = 0*rgb_w
                        bottom = 3*rgb_h
                    elif self.loc_rgb_overlay == 'bottom right':
                        left   = 3*rgb_w
                        bottom = 0*rgb_h
                    elif self.loc_rgb_overlay == 'bottom left':
                        left   = 0*rgb_w
                        bottom = 0*rgb_h
                    else:
                        print ("Invalid location for RGB overlay. Use 'top right', 'top left', 'bottom right', or 'bottom left'.")
                    self.viewport_rgb_render = mujoco.MjrRect(
                        left   = left,
                        bottom = bottom,
                        width  = rgb_w,
                        height = rgb_h,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay).flatten(),
                        depth    = None,
                        viewport = self.viewport_rgb_render,
                        con      = self.ctx,
                    )

                # overlay rgb images
                if self.rgb_overlay_top_right is not None:
                    h_overlay,w_overlay = self.rgb_overlay_top_right.shape[:2]
                    viewport_rgb_top_right = mujoco.MjrRect(
                        left   = 3*w_overlay,
                        bottom = 3*h_overlay,
                        width  = w_overlay,
                        height = h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay_top_right).flatten(),
                        depth    = None,
                        viewport = viewport_rgb_top_right,
                        con      = self.ctx,
                    )
                if self.rgb_overlay_top_left is not None:
                    h_overlay,w_overlay = self.rgb_overlay_top_left.shape[:2]
                    viewport_rgb_top_left = mujoco.MjrRect(
                        left   = 0*w_overlay,
                        bottom = 3*h_overlay,
                        width  = w_overlay,
                        height = h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay_top_left).flatten(),
                        depth    = None,
                        viewport = viewport_rgb_top_left,
                        con      = self.ctx,
                    )
                if self.rgb_overlay_bottom_right is not None:
                    h_overlay,w_overlay = self.rgb_overlay_bottom_right.shape[:2]
                    viewport_rgb_bottom_right = mujoco.MjrRect(
                        left   = 3*w_overlay,
                        bottom = 0*h_overlay,
                        width  = w_overlay,
                        height = h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay_bottom_right).flatten(),
                        depth    = None,
                        viewport = viewport_rgb_bottom_right,
                        con      = self.ctx,
                    )
                if self.rgb_overlay_bottom_left is not None:
                    h_overlay,w_overlay = self.rgb_overlay_bottom_left.shape[:2]
                    viewport_rgb_bottom_left = mujoco.MjrRect(
                        left   = 0*w_overlay,
                        bottom = 0*h_overlay,
                        width  = w_overlay,
                        height = h_overlay,
                    )
                    mujoco.mjr_drawPixels(
                        rgb      = np.flipud(self.rgb_overlay_bottom_left).flatten(),
                        depth    = None,
                        viewport = viewport_rgb_bottom_left,
                        con      = self.ctx,
                    )
                
                # Double buffering
                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + \
                0.1 * (time.time() - render_start)

        if self._paused: # if paused
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []
        
        # clear overlay
        self._overlay.clear()

        # apply perturbation (should this come before mj_step?)
        if self.perturbation:
            self.apply_perturbations()

    def close(self):
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()


class MuJoCoParserClass(object):
    """
        MuJoCo Parser Class 
    """
    def __init__(
            self,
            name          = None,
            rel_xml_path  = None,
            xml_string    = None,
            assets        = None,
            verbose       = True,
        ):
        """ 
            Initialize MuJoCo parser
        """
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.xml_string   = xml_string
        self.assets       = assets
        self.verbose      = verbose
        
        # Constants
        self.tick              = 0
        self.render_tick       = 0
        self.use_mujoco_viewer = False
        
        # Parse xml file
        if (self.rel_xml_path is not None) or (self.xml_string is not None):
            self._parse_xml()
        if self.name is None:
            self.name = self.model_name 

        # Tic-toc
        self.tt = TicTocClass(name='env:[%s]'%(self.name))
        
        # Monitor size
        self.monitor_width,self.monitor_height = get_monitor_size()
            
        # Print
        if self.verbose:
            self.print_info()
            
        # Reset
        self.reset(step=True)
            
    def _parse_xml(self):
        """ 
            Parse xml file
        """
        if self.rel_xml_path is not None:
            self.full_xml_path = os.path.abspath(os.path.join(os.getcwd(),self.rel_xml_path))
            self.model         = mujoco.MjModel.from_xml_path(self.full_xml_path)
        
        if self.xml_string is not None:
            self.model = mujoco.MjModel.from_xml_string(xml=self.xml_string,assets=self.assets)
            
        # Parse xml model name
        parsed_strings = [s for s in self.model.names.split(b'\x00') if s] 
        parsed_strings = [s.decode('utf-8') for s in parsed_strings]
        self.model_name = parsed_strings[0]
        
        self.data             = mujoco.MjData(self.model)
        self.dt               = self.model.opt.timestep
        self.HZ               = int(1/self.dt)
        
        # Integrator (https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtintegrator)
        self.integrator       = self.model.opt.integrator
        if self.integrator == mujoco.mjtIntegrator.mjINT_EULER:
            self.integrator_name = 'EULER'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            self.integrator_name = 'RK4'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT:
            self.integrator_name = 'IMPLICIT'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            self.integrator_name = 'IMPLICITFAST'
        else:
            self.integrator_name = 'UNKNOWN'
        
        # State and action space
        self.n_qpos           = self.model.nq # number of states
        self.n_qvel           = self.model.nv # number of velocities (dimension of tangent space)
        self.n_qacc           = self.model.nv # number of accelerations (dimension of tangent space)
        
        # Geometry
        self.n_geom           = self.model.ngeom # number of geometries
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,geom_idx)
                                 for geom_idx in range(self.model.ngeom)]
        
        # Mesh
        self.n_mesh           = self.model.nmesh # number of meshes
        self.mesh_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_MESH,mesh_idx)
                                 for mesh_idx in range(self.model.nmesh)]
        
        # Body
        self.n_body           = self.model.nbody # number of bodies
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body_idx)
                                 for body_idx in range(self.n_body)]
        self.body_masses      = self.model.body_mass # (kg)
        self.body_total_mass  = self.body_masses.sum()
        
        self.parent_body_names = []
        for b_idx in range(self.n_body):
            parent_id = self.model.body_parentid[b_idx]
            parent_body_name = self.body_names[parent_id]
            self.parent_body_names.append(parent_body_name)
            
        # Degree of Freedom
        self.n_dof            = self.model.nv # degree of freedom (=number of columns of Jacobian)
        self.dof_names        = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_DOF,dof_idx)
                                 for dof_idx in range(self.n_dof)]
        
        # Joint
        self.n_joint          = self.model.njnt # number of joints 
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx)
                                 for joint_idx in range(self.n_joint)]
        self.joint_types      = self.model.jnt_type # joint types
        self.joint_ranges     = self.model.jnt_range # joint ranges
        self.joint_mins       = self.joint_ranges[:,0]
        self.joint_maxs       = self.joint_ranges[:,1]
        
        # Free joint
        self.free_joint_idxs  = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_FREE)[0].astype(np.int32)
        self.free_joint_names = [self.joint_names[joint_idx] for joint_idx in self.free_joint_idxs]
        self.n_free_joint     = len(self.free_joint_idxs)

        # Revolute Joint
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        
        # Prismatic Joint
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.pri_joint_idxs]
        self.n_pri_joint      = len(self.pri_joint_idxs)
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins

        # Revolute + Prismatic Joint Information
        self.n_rev_pri_joint      = self.n_rev_joint + self.n_pri_joint
        self.rev_pri_joint_idxs   = np.concatenate([self.rev_joint_idxs,self.pri_joint_idxs])
        self.rev_pri_joint_names  = self.rev_joint_names + self.pri_joint_names
        self.rev_pri_joint_mins   = np.concatenate([self.rev_joint_mins,self.pri_joint_mins])
        self.rev_pri_joint_maxs   = np.concatenate([self.rev_joint_maxs,self.pri_joint_maxs])
        self.rev_pri_joint_ranges = self.rev_pri_joint_maxs - self.rev_pri_joint_mins
        
        # Controls
        self.n_ctrl           = self.model.nu # number of actuators (or controls)
        self.ctrl_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx) 
                                 for ctrl_idx in range(self.n_ctrl)]
        self.ctrl_ranges      = self.model.actuator_ctrlrange # control range
        self.ctrl_mins        = self.ctrl_ranges[:,0]
        self.ctrl_maxs        = self.ctrl_ranges[:,1]
        self.ctrl_gears       = self.model.actuator_gear[:,0] # gears
        
        # Cameras
        self.n_cam            = self.model.ncam
        self.cam_names        = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_CAMERA,cam_idx) 
                                 for cam_idx in range(self.n_cam)]
        self.cams             = []
        self.cam_fovs         = []
        self.cam_viewports    = []
        for cam_idx in range(self.n_cam):
            cam_name = self.cam_names[cam_idx]
            cam      = mujoco.MjvCamera()
            cam.fixedcamid = self.model.cam(cam_name).id
            cam.type       = mujoco.mjtCamera.mjCAMERA_FIXED
            cam_fov        = self.model.cam_fovy[cam_idx]
            viewport       = mujoco.MjrRect(0,0,800,600) # SVGA?
            # Append
            self.cams.append(cam)
            self.cam_fovs.append(cam_fov)
            self.cam_viewports.append(viewport)
            
        # qpos and qvel indices attached to the controls
        """ 
        # Usage
        self.env.data.qpos[self.env.ctrl_qpos_idxs] # joint position
        self.env.data.qvel[self.env.ctrl_qvel_idxs] # joint velocity
        """
        self.ctrl_qpos_idxs = []
        self.ctrl_qpos_names = []
        self.ctrl_qpos_mins = []
        self.ctrl_qpos_maxs = []
        self.ctrl_qvel_idxs = []
        self.ctrl_types = []
        for ctrl_idx in range(self.n_ctrl):
            # transmission (joint) index attached to an actuator, we assume that there is just one joint attached
            joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0] 
            # joint position attached to control
            self.ctrl_qpos_idxs.append(self.model.jnt_qposadr[joint_idx])
            self.ctrl_qpos_names.append(self.joint_names[joint_idx])
            self.ctrl_qpos_mins.append(self.joint_ranges[joint_idx,0])
            self.ctrl_qpos_maxs.append(self.joint_ranges[joint_idx,1])
            # joint velocity attached to control
            self.ctrl_qvel_idxs.append(self.model.jnt_dofadr[joint_idx])
            # Check types
            trntype = self.model.actuator_trntype[ctrl_idx]
            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                self.ctrl_types.append('JOINT')
            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
                self.ctrl_types.append('TENDON')
            else:
                self.ctrl_types.append('UNKNOWN(trntype=%d)'%(trntype))
                
        # Sensor
        self.n_sensor         = self.model.nsensor
        self.sensor_names     = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SENSOR,sensor_idx)
                                 for sensor_idx in range(self.n_sensor)]
        
        # Site
        self.n_site           = self.model.nsite
        self.site_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,site_idx)
                                 for site_idx in range(self.n_site)]
        
    def print_info(self):
        """ 
            Print model information
        """
        print ("")
        print ("-----------------------------------------------------------------------------")
        print ("name:[%s] dt:[%.3f] HZ:[%d]"%(self.name,self.dt,self.HZ))
        print (" n_qpos:[%d] n_qvel:[%d] n_qacc:[%d] n_ctrl:[%d]"%(self.n_qpos,self.n_qvel,self.n_qacc,self.n_ctrl))
        print (" integrator:[%s]"%(self.integrator_name))

        print ("")
        print ("n_body:[%d]"%(self.n_body))
        for body_idx,body_name in enumerate(self.body_names):
            body_mass = self.body_masses[body_idx]
            print (" [%d/%d] [%s] mass:[%.2f]kg"%(body_idx,self.n_body,body_name,body_mass))
        print ("body_total_mass:[%.2f]kg"%(self.body_total_mass))
        
        print ("")
        print ("n_geom:[%d]"%(self.n_geom))
        print ("geom_names:%s"%(self.geom_names))

        print ("")
        print ("n_mesh:[%d]"%(self.n_mesh))
        print ("mesh_names:%s"%(self.mesh_names))

        print ("")
        print ("n_joint:[%d]"%(self.n_joint))
        for joint_idx,joint_name in enumerate(self.joint_names):
            print (" [%d/%d] [%s] axis:%s"%
                   (joint_idx,self.n_joint,joint_name,self.model.joint(joint_idx).axis))
        # print ("joint_types:[%s]"%(self.joint_types))
        # print ("joint_ranges:[%s]"%(self.joint_ranges))

        print ("")
        print ("n_dof:[%d] (=number of rows of Jacobian)"%(self.n_dof))
        for dof_idx,dof_name in enumerate(self.dof_names):
            joint_name= self.joint_names[self.model.dof_jntid[dof_idx]]
            body_name= self.body_names[self.model.dof_bodyid[dof_idx]]
            print (" [%d/%d] [%s] attached joint:[%s] body:[%s]"%
                   (dof_idx,self.n_dof,dof_name,joint_name,body_name))
        
        print ("\nFree joint information. n_free_joint:[%d]"%(self.n_free_joint))
        for idx,free_joint_name in enumerate(self.free_joint_names):
            body_name_attached = self.body_names[self.model.joint(self.free_joint_idxs[idx]).bodyid[0]]
            print (" [%d/%d] [%s] body_name_attached:[%s]"%
                   (idx,self.n_free_joint,free_joint_name,body_name_attached))
            
        print ("\nRevolute joint information. n_rev_joint:[%d]"%(self.n_rev_joint))
        for idx,rev_joint_name in enumerate(self.rev_joint_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f]"%
                   (idx,self.n_rev_joint,rev_joint_name,self.rev_joint_mins[idx],self.rev_joint_maxs[idx]))

        print ("\nPrismatic joint information. n_pri_joint:[%d]"%(self.n_pri_joint))
        for idx,pri_joint_name in enumerate(self.pri_joint_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f]"%
                   (idx,self.n_pri_joint,pri_joint_name,self.pri_joint_mins[idx],self.pri_joint_maxs[idx]))
            
        print ("\nControl information. n_ctrl:[%d]"%(self.n_ctrl))
        for idx,ctrl_name in enumerate(self.ctrl_names):
            print (" [%d/%d] [%s] range:[%.3f]~[%.3f] gear:[%.2f] type:[%s]"%
                   (idx,self.n_ctrl,ctrl_name,self.ctrl_mins[idx],self.ctrl_maxs[idx],
                    self.ctrl_gears[idx],self.ctrl_types[idx]))
            
        print ("\nCamera information. n_cam:[%d]"%(self.n_cam))
        for idx,cam_name in enumerate(self.cam_names):
            print (" [%d/%d] [%s] fov:[%.1f]"%
                   (idx,self.n_cam,cam_name,self.cam_fovs[idx]))
            
        print ("")
        print ("n_sensor:[%d]"%(self.n_sensor))
        print ("sensor_names:%s"%(self.sensor_names))
        print ("n_site:[%d]"%(self.n_site))
        print ("site_names:%s"%(self.site_names))
        print ("-----------------------------------------------------------------------------")
        
    def print_body_joint_info(self):
        """ 
            Print body and joint information (with more details)
        """
        from termcolor import colored
        # Summarize kinematic chain information
        JOINT_TYPE_MAP = {
            mujoco.mjtJoint.mjJNT_FREE: 'free',
            mujoco.mjtJoint.mjJNT_HINGE: 'revolute',
            mujoco.mjtJoint.mjJNT_SLIDE: 'prismatic',
        }
        for body_idx in range(self.n_body):
            # Parse body information
            body_name = self.body_names[body_idx] # body name
            body = self.model.body(body_name) # mujoco body object
            parent_body_name = self.body_names[body.parentid[0]]
            p_body_offset,quat_body_offset = body.pos,body.quat # body offset
            T_body_offset = pr2t(p_body_offset,quat2r(quat_body_offset)) # [4x4]
            print ("[%2d/%d] body_name:[%s] parent_body_name:[%s]"%
                (body_idx,self.n_body,colored(body_name,'green'),colored(parent_body_name,'green')))
            print (" body p_offset:[%.2f,%.2f,%.2f] quat_offset:[%.2f,%.2f,%.2f,%.2f]"%
                (p_body_offset[0],p_body_offset[1],p_body_offset[2],
                    quat_body_offset[0],quat_body_offset[1],quat_body_offset[2],quat_body_offset[3]))
            # Parse joint information
            n_joint = body.jntnum # number of attached joints 
            if n_joint == 0: # fixed joint
                print (" n_joint:[0] (%s)"%(colored('this body has no joint','blue')))
            elif n_joint == 1: # one moving joint (revolute or prismatic)
                joint = self.model.joint(body.jntadr[0]) # joint attached joint
                joint_name = joint.name
                joint_type = JOINT_TYPE_MAP[joint.type[0]]
                p_joint_offset,joint_axis = joint.pos,joint.axis
                print (" joint_name:[%s] n_joint:%s type:[%s]"%
                    (colored(joint_name,'green'),n_joint,colored('%s'%(joint_type),'green')))
                print (" joint p_offset:[%.2f,%.2f,%.2f] axis:[%.1f,%.1f,%.1f]"%
                    (p_joint_offset[0],p_joint_offset[1],p_joint_offset[2],
                        joint_axis[0],joint_axis[1],joint_axis[2]))
            else: # composite joints (not supported)
                print (" n_joint:%s (%s)"%
                    (n_joint,colored('composite joints','red')))
            print ("")
        
    def reset(self,step=True):
        """
            Reset
        """
        time.sleep(1e-3) # add some sleep?
        mujoco.mj_resetData(self.model,self.data) # reset data
        
        if step:
            mujoco.mj_step(self.model,self.data)
            # mujoco.mj_forward(self.model,self.data) # forward <= is this necessary?
        
        # Reset ticks
        self.tick        = 0
        self.render_tick = 0
        # Reset wall time
        self.init_sim_time    = self.data.time
        self.init_wall_time   = time.time()
        self.accum_wall_time  = 0.0  # 누적 wall 시간
        self.last_wall_update = time.time()
        # Others
        self.xyz_left_double_click = None 
        self.xyz_right_double_click = None 
        # Print
        if self.verbose: print ("env:[%s] reset"%(self.name))
        
    def init_viewer(
            self,
            title             = None,
            fullscreen        = False,
            width             = 1400,
            height            = 1000,
            hide_menu         = True,
            fontscale         = mujoco.mjtFontScale.mjFONTSCALE_200.value,
            azimuth           = 170, # None,
            distance          = 5.0, # None,
            elevation         = -20, # None,
            lookat            = [0.01,0.11,0.5], # None,
            transparent       = None,
            contactpoint      = None,
            contactwidth      = None,
            contactheight     = None,
            contactrgba       = None,
            joint             = None,
            jointlength       = None,
            jointwidth        = None,
            jointrgba         = None,
            geomgroup_0       = None, # floor sky
            geomgroup_1       = None, # collision
            geomgroup_2       = None, # visual
            geomgroup_3       = None,
            geomgroup_4       = None,
            geomgroup_5       = None,
            update            = False,
            maxgeom           = 50000,
            perturbation      = True,
            black_sky         = False,
            convex_hull       = None,
            n_fig             = 0,
            use_rgb_overlay   = False,
            loc_rgb_overlay   = 'top right',
            pre_render        = False,
        ):
        """
        Initialize the MuJoCo viewer with the given parameters.
        
        Parameters:
            title (str): Viewer window title.
            fullscreen (bool): Whether to use fullscreen mode.
            width (int): Width of the viewer window.
            height (int): Height of the viewer window.
            hide_menu (bool): Whether to hide the viewer menu.
            fontscale: Font scaling factor.
            azimuth (float): Initial camera azimuth angle.
            distance (float): Initial camera distance.
            elevation (float): Initial camera elevation.
            lookat (list or np.array): Initial camera look-at position.
            transparent, contactpoint, contactwidth, contactheight, contactrgba:
                Parameters for contact point visualization.
            joint, jointlength, jointwidth, jointrgba:
                Parameters for joint visualization.
            geomgroup_0 ~ geomgroup_5: Geometry group visibility flags.
            update (bool): Whether to immediately update the viewer.
            maxgeom (int): Maximum number of geometries.
            perturbation (bool): Whether to allow perturbation.
            black_sky (bool): Whether to render a black skybox.
            convex_hull: Flag for convex hull visualization.
            n_fig (int): Number of figures for overlay plotting.
            use_rgb_overlay (bool): Whether to use an RGB overlay.
            loc_rgb_overlay (str): Location of the first RGB overlay.
            pre_render (bool): Whether to perform an initial render.
        
        Returns:
            None
        """
        self.use_mujoco_viewer = True
        if title is None: title = self.name
        
        # Fullscreen (this overrides 'width' and 'height')
        w_monitor,h_monitor = get_monitor_size()
        if fullscreen:
            width,height = w_monitor,h_monitor
            
        if width <= 1.0 and height <= 1.0:
            width = int(width*w_monitor)
            height = int(height*h_monitor)

        time.sleep(1e-3)
        self.viewer = MuJoCoMinimalViewer(
            self.model,
            self.data,
            mode              = 'window',
            title             = title,
            width             = width,
            height            = height,
            hide_menus        = hide_menu,
            maxgeom           = maxgeom,
            perturbation      = perturbation,
            n_fig             = n_fig,
            use_rgb_overlay   = use_rgb_overlay,
            loc_rgb_overlay   = loc_rgb_overlay,
        )
        self.viewer.ctx = mujoco.MjrContext(self.model,fontscale)
        
        # Set viewer
        self.set_viewer(
            azimuth       = azimuth,
            distance      = distance,
            elevation     = elevation,
            lookat        = lookat,
            transparent   = transparent,
            contactpoint  = contactpoint,
            contactwidth  = contactwidth,
            contactheight = contactheight,
            contactrgba   = contactrgba,
            joint         = joint,
            jointlength   = jointlength,
            jointwidth    = jointwidth,
            jointrgba     = jointrgba,
            geomgroup_0   = geomgroup_0,
            geomgroup_1   = geomgroup_1,
            geomgroup_2   = geomgroup_2,
            geomgroup_3   = geomgroup_3,
            geomgroup_4   = geomgroup_4,
            geomgroup_5   = geomgroup_5,
            black_sky     = black_sky,
            convex_hull   = convex_hull,
            update        = update,
        )
        if pre_render: self.render()
        # Print
        if self.verbose: print ("env:[%s] initalize viewer"%(self.name))
        
    def set_viewer(
            self,
            azimuth       = None,
            distance      = None,
            elevation     = None,
            lookat        = None,
            transparent   = None,
            contactpoint  = None,
            contactwidth  = None,
            contactheight = None,
            contactrgba   = None,
            joint         = None,
            jointlength   = None,
            jointwidth    = None,
            jointrgba     = None,
            geomgroup_0   = None,
            geomgroup_1   = None,
            geomgroup_2   = None,
            geomgroup_3   = None,
            geomgroup_4   = None,
            geomgroup_5   = None,
            black_sky     = None,
            convex_hull   = None,
            update        = False,
        ):
        """
        Set or update the viewer’s camera and visualization parameters.
        
        Parameters:
            azimuth (float): Camera azimuth angle.
            distance (float): Camera distance.
            elevation (float): Camera elevation angle.
            lookat (list or np.array): Camera look-at position.
            transparent (bool): Flag for making dynamic geometries transparent.
            contactpoint (bool): Flag for displaying contact points.
            contactwidth (float): Contact point width.
            contactheight (float): Contact point height.
            contactrgba (list): RGBA color for contact points.
            joint (bool): Flag for joint visualization.
            jointlength (float): Length of joint visuals.
            jointwidth (float): Width of joint visuals.
            jointrgba (list): RGBA color for joints.
            geomgroup_0 ~ geomgroup_5: Visibility flags for geometry groups.
            black_sky (bool): Flag for enabling/disabling skybox.
            convex_hull (bool): Flag for convex hull visualization.
            update (bool): If True, perform an immediate update.
        
        Returns:
            None
        """
        # Basic viewer setting (azimuth, distance, elevation, and lookat)
        if azimuth is not None: self.viewer.cam.azimuth = azimuth
        if distance is not None: self.viewer.cam.distance = distance
        if elevation is not None: self.viewer.cam.elevation = elevation
        if lookat is not None: self.viewer.cam.lookat = lookat
        # Make dynamic geoms more transparent
        if transparent is not None: 
            self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
        # Contact point
        if contactpoint is not None: self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contactpoint
        if contactwidth is not None: self.model.vis.scale.contactwidth = contactwidth
        if contactheight is not None: self.model.vis.scale.contactheight = contactheight
        if contactrgba is not None: self.model.vis.rgba.contactpoint = contactrgba
        # Joint
        if joint is not None: self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = joint
        if jointlength is not None: self.model.vis.scale.jointlength = jointlength
        if jointwidth is not None: self.model.vis.scale.jointwidth = jointwidth
        if jointrgba is not None: self.model.vis.rgba.joint = jointrgba
        # Geom group
        if geomgroup_0 is not None: self.viewer.vopt.geomgroup[0] = geomgroup_0
        if geomgroup_1 is not None: self.viewer.vopt.geomgroup[1] = geomgroup_1
        if geomgroup_2 is not None: self.viewer.vopt.geomgroup[2] = geomgroup_2
        if geomgroup_3 is not None: self.viewer.vopt.geomgroup[3] = geomgroup_3
        if geomgroup_4 is not None: self.viewer.vopt.geomgroup[4] = geomgroup_4
        if geomgroup_5 is not None: self.viewer.vopt.geomgroup[5] = geomgroup_5
        # Skybox
        if black_sky is not None: self.viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = not black_sky
        # Convex hull
        if convex_hull is not None: self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = convex_hull
        # Render to update settings
        if update:
            mujoco.mj_forward(self.model,self.data) 
            mujoco.mjv_updateScene(
                self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
            mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
            
    def get_viewer_cam_info(self,verbose=False):
        """
        Retrieve the current camera parameters from the viewer.
        
        Parameters:
            verbose (bool): If True, print camera info.
        
        Returns:
            tuple: (azimuth, distance, elevation, lookat) of the camera.
        """
        azimuth   = self.viewer.cam.azimuth
        distance  = self.viewer.cam.distance
        elevation = self.viewer.cam.elevation
        lookat    = self.viewer.cam.lookat.copy()
        if verbose:
            print ("azimuth:[%.2f] distance:[%.2f] elevation:[%.2f] lookat:%s]"%
                   (azimuth,distance,elevation,lookat))
        return azimuth,distance,elevation,lookat
    
    def is_viewer_alive(self):
        """
        Check whether the viewer window is still active.
        
        Returns:
            bool: True if the viewer is alive, False otherwise.
        """
        return self.viewer.is_alive
    
    def close_viewer(self):
        """
        Close and clean up the viewer resources.
        
        Returns:
            None
        """
        self.use_mujoco_viewer = False
        self.viewer.close()

    def viewer_text_overlay(
            self,
            loc   = 'bottom left',
            text1 = '',
            text2 = '',
        ):
        """ 
            Add text overlay on the viewer
            Parameters:
                loc: ['top','top right','top left','bottom','bottom right','bottom left']
                text1: string
                text2: string
        """
        self.viewer.add_overlay(loc=loc,text1=text1,text2=text2)

    def viewer_rgb_overlay(
            self,
            rgb = None,
            loc = 'top right',
        ):
        """ 
            Add RGB overlay on the viewer
            Parameters:
                loc: ['top','top right','top left','bottom','bottom right','bottom left']
                rgb: RGB image
        """
        self.viewer.plot_rgb_overlay(rgb=rgb,loc=loc)

    def render(self):
        """
        Render the current simulation state to the viewer.
        
        Returns:
            None
        """
        if self.use_mujoco_viewer:
            self.viewer.render()
        else:
            print ("[%s] Viewer NOT initialized."%(self.name))
            
    def loop_every(self,HZ=None,tick_every=None):
        """
        Determine if the simulation loop should execute based on simulation tick or frequency.
        
        Parameters:
            HZ (float): Frequency in Hz to trigger the loop.
            tick_every (int): Number of simulation ticks between executions.
        
        Returns:
            bool: True if the loop should execute, False otherwise.
        """
        # tick = int(self.get_sim_time()/self.dt)
        FLAG = False
        if HZ is not None:
            FLAG = (self.tick-1)%(int(1/self.dt/HZ))==0
        if tick_every is not None:
            FLAG = (self.tick-1)%(tick_every)==0
        return FLAG
    
    def step(
            self,
            ctrl             = None,
            ctrl_idxs        = None,
            ctrl_names       = None,
            joint_names      = None,
            nstep            = 1,
            increase_tick    = True,
            step_flag        = True,
        ):
        """
        Advance the simulation by a specified number of steps, optionally applying control inputs.
        
        Parameters:
            ctrl (np.array): Control inputs to apply.
            ctrl_idxs (list): Indices of controls to update.
            ctrl_names (list): Names of controls to update.
            joint_names (list): Names of joints corresponding to controls.
            nstep (int): Number of simulation steps to execute.
            increase_tick (bool): Whether to increment the simulation tick counter.
        
        Returns:
            None
        """
        if step_flag:
            if ctrl is not None:
                if ctrl_names is not None: # when given 'ctrl_names' explicitly
                    ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)
                elif joint_names is not None: # when given 'joint_names' explicitly
                    ctrl_idxs = self.get_idxs_step(joint_names=joint_names)            
                # Apply control
                if ctrl_idxs is None: 
                    self.data.ctrl[:] = ctrl
                else: 
                    self.data.ctrl[ctrl_idxs] = ctrl
            mujoco.mj_step(self.model,self.data,nstep=nstep)
        
        # Update wall time (conditioned on 'step_flag')
        self.increase_wall_time(step_flag=step_flag)

        # Increase tick
        if increase_tick: self.increase_tick()

    def forward(self,q=None,joint_idxs=None,joint_names=None,increase_tick=True):
        """
        Compute the forward kinematics of the system, optionally updating joint positions.
        
        Parameters:
            q (np.array): Joint positions to set.
            joint_idxs (list): Indices of joints to update.
            joint_names (list): Names of joints to update.
            increase_tick (bool): Whether to increment the simulation tick counter.
        
        Returns:
            None
        """
        if q is not None:
            if joint_names is not None: # if 'joint_names' is not None, it override 'joint_idxs'
                joint_idxs = self.get_idxs_fwd(joint_names=joint_names)
            if joint_idxs is not None: 
                self.data.qpos[joint_idxs] = q
            else: self.data.qpos = q
        mujoco.mj_forward(self.model,self.data)
        if increase_tick: 
            self.increase_tick()

    def increase_wall_time(self,step_flag=True):
        """
        Increment the accumulated wall time.
        """
        current_wall_time = time.time()
        if step_flag:
            # Increment wall time only when 'step_flag' is True
            self.accum_wall_time += current_wall_time - self.last_wall_update
        self.last_wall_update = current_wall_time

    def increase_tick(self):
        """
        Increment the simulation tick counter.
        
        Returns:
            None
        """
        self.tick = self.tick + 1

    def get_state(self):
        """ 
        Retrieve the current simulation state including time, joint positions, velocities, and actuator states.
        
        Returns:
            dict: A dictionary containing state information.

        The state vector in MuJoCo is:
            x = (mjData.time, mjData.qpos, mjData.qvel, mjData.act)
        Next we turn to the controls and applied forces. The control vector in MuJoCo is
            u = (mjData.ctrl, mjData.qfrc_applied, mjData.xfrc_applied)
        These quantities specify control signals (mjData.ctrl) for the actuators defined in the model, 
        or directly apply forces and torques specified in joint space (mjData.qfrc_applied) 
        or in Cartesian space (mjData.xfrc_applied).
        """
        state = {
            'tick':self.tick,
            'time':self.data.time,
            'qpos':self.data.qpos.copy(), # [self.model.nq]
            'qvel':self.data.qvel.copy(), # [self.model.nv]
            'act':self.data.act.copy(),
        }
        return state
    
    def store_state(self):
        """
        Store the current simulation state for later restoration.
        
        Returns:
            None
        """
        state = self.get_state()
        self.state_stored = copy.deepcopy(state) # deep copy
        
    def restore_state(self):
        """
        Restore the simulation state from a previously stored state.
        
        Returns:
            None
        """
        state = self.state_stored
        self.set_state(
            qpos = state['qpos'],
            qvel = state['qvel'],
            act  = state['act'],
        )
        mujoco.mj_forward(self.model,self.data)
        
    def set_state(
            self,
            tick = None,
            time = None,
            qpos = None,
            qvel = None,
            act  = None, # used for simulating tendons and muscles
            ctrl = None,
            step = False
        ):
        """
        Set the simulation state with provided values.
        
        Parameters:
            tick (int): Simulation tick.
            time (float): Simulation time.
            qpos (np.array): Joint positions.
            qvel (np.array): Joint velocities.
            act (np.array): Actuator states.
            ctrl (np.array): Control signals.
            step (bool): If True, perform a simulation step after setting the state.
        
        Returns:
            None
        """
        if tick is not None: self.tick = tick
        if time is not None: self.data.time = time
        if qpos is not None: self.data.qpos = qpos.copy()
        if qvel is not None: self.data.qvel = qvel.copy()
        if act is not None: self.data.act = act.copy()
        if ctrl is not None: self.data.ctrl = ctrl.copy()
        # Forward dynamics
        if step: 
            mujoco.mj_step(self.model,self.data)
            
    def solve_inverse_dynamics(self,qacc=None):
        """
        Solve inverse dynamics to compute the forces required to achieve the given joint accelerations.
        
        Parameters:
            qacc (np.array): Desired joint accelerations.
        
        Returns:
            np.array: The computed inverse dynamics forces.
        """
        if qacc is None:
            qacc = np.zeros(self.n_qacc)
        # Set desired qacc
        self.data.qacc = qacc.copy()
        # Store state
        self.store_state()
        # Solve inverse dynamics
        mujoco.mj_inverse(self.model,self.data)
        # Restore state
        self.restore_state()
        # Return  
        """
            Output is 'qfrc_inverse'
            This is the force that must have acted on the system in order to achieve the observed acceleration 'mjData.qacc'.
        """
        qfrc_inverse = self.data.qfrc_inverse # [n_qacc]
        return qfrc_inverse.copy()
    
    def set_p_base_body(self,body_name='base',p=np.array([0,0,0]),forward=True):
        """
        Set the position of the base body.
        
        Parameters:
            body_name (str): Name of the base body.
            p (np.array): New position (3D vector).
        
        Returns:
            None
        """
        jntadr  = self.model.body(body_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr:qposadr+3] = p
        if forward:
            mujoco.mj_forward(self.model,self.data)
        
    def set_R_base_body(self,body_name='base',R=rpy2r(np.radians([0,0,0]))):
        """
        Set the orientation of the base body.
        
        Parameters:
            body_name (str): Name of the base body.
            R (np.array): New rotation matrix (3x3).
        
        Returns:
            None
        """
        jntadr  = self.model.body(body_name).jntadr[0]
        qposadr = self.model.jnt_qposadr[jntadr]
        self.data.qpos[qposadr+3:qposadr+7] = r2quat(R)
        mujoco.mj_forward(self.model,self.data)

    def set_pR_base_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),T=None):
        """
        Set the pose (position and rotation) of the base body.
        
        Parameters:
            body_name (str): Name of the base body.
            p (np.array): New position (3D vector).
            R (np.array): New rotation matrix (3x3).
            T (np.array): Transformation matrix (4x4) that overrides p and R if provided.
        
        Returns:
            None
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        self.set_p_base_body(body_name=body_name,p=p)
        self.set_R_base_body(body_name=body_name,R=R)
                
    def set_T_base_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),T=None):
        """
        Set the pose of the base body using a transformation matrix.
        
        Parameters:
            body_name (str): Name of the base body.
            p (np.array): Position vector.
            R (np.array): Rotation matrix.
            T (np.array): Transformation matrix that, if provided, overrides p and R.
        
        Returns:
            None
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        self.set_p_base_body(body_name=body_name,p=p)
        self.set_R_base_body(body_name=body_name,R=R)
        
    def set_p_body(self,body_name='base',p=np.array([0,0,0]),forward=True):
        """
        Set the position of a specified body.
        
        Parameters:
            body_name (str): Name of the body.
            p (np.array): New position (3D vector).
            forward (bool): If True, perform forward kinematics after setting.
        
        Returns:
            None
        """
        self.model.body(body_name).pos = p
        if forward: self.forward(increase_tick=False)
        
    def set_R_body(self,body_name='base',R=np.eye(3),forward=True):
        """
        Set the orientation of a specified body.
        
        Parameters:
            body_name (str): Name of the body.
            R (np.array): New rotation matrix (3x3).
            forward (bool): If True, perform forward kinematics after setting.
        
        Returns:
            None
        """
        self.model.body(body_name).quat = r2quat(R)
        if forward: self.forward(increase_tick=False)
        
    def set_pR_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),forward=True):
        """
        Set both the position and orientation of a specified body.
        
        Parameters:
            body_name (str): Name of the body.
            p (np.array): New position.
            R (np.array): New rotation matrix.
            forward (bool): Whether to perform forward kinematics after updating.
        
        Returns:
            None
        """
        self.model.body(body_name).pos = p
        self.model.body(body_name).quat = r2quat(R)
        if forward: self.forward(increase_tick=False)

    def set_T_body(self,body_name='base',p=np.array([0,0,0]),R=np.eye(3),T=None,forward=True):
        """
        Set both the position and orientation of a specified body.
        
        Parameters:
            body_name (str): Name of the body.
            p (np.array): New position.
            R (np.array): New rotation matrix.
            forward (bool): Whether to perform forward kinematics after updating.
        
        Returns:
            None
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        self.model.body(body_name).pos = p
        self.model.body(body_name).quat = r2quat(R)
        if forward: self.forward(increase_tick=False)        
        
    def set_p_mocap(self,mocap_name='',p=np.array([0,0,0])):
        """
        Set the position of a mocap body.
        
        Parameters:
            mocap_name (str): Name of the mocap body.
            p (np.array): New position.
        
        Returns:
            None
        """
        mocap_idx = self.model.body_mocapid[self.body_names.index(mocap_name)]
        self.data.mocap_pos[mocap_idx] = p
        
    def set_R_mocap(self,mocap_name='',R=np.eye(3)):
        """
        Set the orientation of a mocap body.
        
        Parameters:
            mocap_name (str): Name of the mocap body.
            R (np.array): New rotation matrix.
        
        Returns:
            None
        """
        mocap_idx = self.model.body_mocapid[self.body_names.index(mocap_name)]
        self.data.mocap_quat[mocap_idx] = r2quat(R)

    def set_pR_mocap(self,mocap_name='',p=np.array([0,0,0]),R=np.eye(3)):
        """
        Set the pose of a mocap body.
        
        Parameters:
            mocap_name (str): Name of the mocap body.
            p (np.array): New position.
            R (np.array): New rotation matrix.
        
        Returns:
            None
        """
        self.set_p_mocap(mocap_name=mocap_name,p=p)
        self.set_R_mocap(mocap_name=mocap_name,R=R)
        
    def set_geom_color(
            self,
            body_names_to_color   = None,
            body_names_to_exclude = ['world'],
            body_names_to_exclude_including = [],
            rgba                  = [0.75,0.95,0.15,1.0],
            rgba_list             = None,
        ):
        """
        Set the color for geometries attached to specified bodies.
        
        Parameters:
            body_names_to_color (list): List of body names to color.
            body_names_to_exclude (list): Bodies to exclude.
            body_names_to_exclude_including (list): Bodies containing these substrings are excluded.
            rgba (list): Default RGBA color.
            rgba_list (list): List of RGBA colors corresponding to each body.
        
        Returns:
            None
        """
        def should_exclude(x, exclude_list):
            for exclude in exclude_list:
                if exclude in x:
                    return True
            return False
        
        if body_names_to_color is None: # default is to color all geometries
            body_names_to_color = self.body_names
        for idx,body_name in enumerate(body_names_to_color): # for all bodies
            if body_name in body_names_to_exclude: # exclude specific bodies
                continue 
            if should_exclude(body_name,body_names_to_exclude_including): 
                # exclude body_name including ones in 'body_names_to_exclude_including'
                continue
            body_idx = self.body_names.index(body_name)
            geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx]
            for geom_idx in geom_idxs: # for geoms attached to the body
                if rgba_list is None:
                    self.model.geom(geom_idx).rgba = rgba
                else:
                    self.model.geom(geom_idx).rgba = rgba_list[idx]
                    
    def set_geom_alpha(self,alpha=1.0,body_names_to_exclude=['world']):
        """
        Set the transparency (alpha) value for geometries.
        
        Parameters:
            alpha (float): Transparency value (0 to 1).
            body_names_to_exclude (list): Bodies to exclude from this change.
        
        Returns:
            None
        """
        for g_idx in range(self.n_geom): # for each geom
            geom = self.model.geom(g_idx)
            body_name = self.body_names[geom.bodyid[0]]
            if body_name in body_names_to_exclude: continue # exclude certain bodies
            # Change geom alpha
            self.model.geom(g_idx).rgba[3] = alpha
            
    def get_sim_time(self,init_flag=False):
        """
        Get the elapsed simulation time since initialization.
        
        Parameters:
            init_flag (bool): If True, reset the simulation time reference.
        
        Returns:
            float: Elapsed simulation time in seconds.
        """
        if init_flag:
            self.init_sim_time = self.data.time
        elapsed_time = self.data.time - self.init_sim_time
        return elapsed_time
    
    def reset_sim_time(self):
        """
        Reset the simulation time reference.
        
        Returns:
            None
        """
        self.init_sim_time = self.data.time
        
    def reset_wall_time(self):
        """
        Reset the wall-clock time reference.
        
        Returns:
            None
        """
        self.init_wall_time = time.time()
        
    def get_wall_time(self,init_flag=False):
        """
        Get the elapsed wall-clock time since the last reset.
        
        Parameters:
            init_flag (bool): If True, reset the wall clock reference.
        
        Returns:
            float: Elapsed wall time in seconds.
        """
        if init_flag:
            self.accum_wall_time = 0.0
            self.last_wall_update = time.time()
        return self.accum_wall_time
    
    def grab_rgbd_img(self):
        """
        Capture an RGB-D image (color and depth) from the current viewer.
        
        Returns:
            tuple: (rgb_img, depth_img) where rgb_img is a uint8 image and depth_img is a float32 image.
        """
        rgb_img   = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img) # flip up-down

        # Rescale depth image
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1 - depth_img * (1 - near / far))
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img
    
    def get_T_viewer(self):
        """
        Compute and return the current transformation matrix of the viewer camera.
        
        Returns:
            np.array: A 4x4 transformation matrix.
        """
        cam_lookat    = self.viewer.cam.lookat
        cam_elevation = self.viewer.cam.elevation
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance

        p_lookat = cam_lookat
        R_lookat = rpy2r(np.deg2rad([0,-cam_elevation,cam_azimuth]))
        T_lookat = pr2t(p_lookat,R_lookat)
        T_viewer = T_lookat @ pr2t(np.array([-cam_distance,0,0]),np.eye(3))
        return T_viewer
    
    def get_pcd_from_depth_img(self,depth_img,fovy=45):
        """
        Generate point cloud data from a given depth image.
        
        Parameters:
            depth_img (np.array): The depth image.
            fovy (float): Field of view in the y-direction.
        
        Returns:
            tuple: (pcd, xyz_img, xyz_img_world) representing point cloud and intermediate coordinate arrays.
        """
        # Get camera pose
        T_viewer = self.get_T_viewer()

        # Camera intrinsic
        img_height = depth_img.shape[0]
        img_width = depth_img.shape[1]
        focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)
        cam_matrix = np.array(((focal_scaling,0,img_width/2),
                            (0,focal_scaling,img_height/2),
                            (0,0,1)))

        # Estimate 3D point from depth image
        xyz_img = meters2xyz(depth_img,cam_matrix) # [H x W x 3]
        xyz_transpose = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
        xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]

        # To world coordinate
        xyzone_world_transpose = T_viewer @ xyzone_transpose
        xyz_world_transpose = xyzone_world_transpose[:3,:] # [3 x N]
        xyz_world = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]

        xyz_img_world = xyz_world.reshape(depth_img.shape[0],depth_img.shape[1],3)

        return xyz_world,xyz_img,xyz_img_world
    
    def get_egocentric_rgb(
            self,
            p_ego        = None,
            p_trgt       = None,
            rsz_rate     = None,
            fovy         = None,
            restore_view = True,
        ):
        """
        Capture an egocentric RGB image based on the provided ego and target positions.
        
        Parameters:
            p_ego (np.array): Position of the ego camera.
            p_trgt (np.array): Target position.
            rsz_rate (float): Resize rate for the captured image.
            fovy (float): Field of view angle.
            restore_view (bool): Whether to restore the original camera view after capturing.
        
        Returns:
            np.array: The captured RGB image.
        """
        if restore_view:
            # Backup camera information
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.get_viewer_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = compute_view_params(
                camera_pos = p_ego,
                target_pos = p_trgt,
                up_vector  = np.array([0,0,1]),
            )
            self.set_viewer(
                azimuth   = cam_azimuth,
                distance  = cam_distance,
                elevation = cam_elevation,
                lookat    = cam_lookat,
                update    = True,
            )
        
        # Grab RGB and depth image
        rgb_img,_ = self.grab_rgbd_img() # get rgb and depth images

        # Resize rgb_image and depth_img (optional)
        if rsz_rate is not None:
            h = int(rgb_img.shape[0]*rsz_rate)
            w = int(rgb_img.shape[1]*rsz_rate)
            rgb_img = cv2.resize(rgb_img,(w,h),interpolation=cv2.INTER_NEAREST)
            
        # Restore view
        if restore_view:
            # Restore camera information
            self.set_viewer(
                azimuth   = viewer_azimuth,
                distance  = viewer_distance,
                elevation = viewer_elevation,
                lookat    = viewer_lookat,
                update    = True,
            )
        return rgb_img
    
    def get_egocentric_rgbd_pcd(
            self,
            p_ego            = None,
            p_trgt           = None,
            rsz_rate_for_pcd = None,
            rsz_rate_for_img = None,
            fovy             = None,
            restore_view     = True,
        ):
        """
        Capture egocentric RGB, depth images and generate point cloud data.
        
        Parameters:
            p_ego (np.array): Ego camera position.
            p_trgt (np.array): Target position.
            rsz_rate_for_pcd (float): Resize rate for point cloud generation.
            rsz_rate_for_img (float): Resize rate for images.
            fovy (float): Field of view.
            restore_view (bool): Whether to restore the original camera view.
        
        Returns:
            tuple: (rgb_img, depth_img, pcd, xyz_img, xyz_img_world)

        Get egocentric 1) RGB image, 2) Depth image, 3) Point Cloud Data
        return: (rgb_img,depth_img,pcd,xyz_img,xyz_img_world)
        THIS FUNCTION CAN BE PROBLEMETIC as it cannot control the twist around the line of sight
        (see https://mujoco.readthedocs.io/en/stable/programming/visualization.html for more details
        )
        """
        if restore_view:
            # Backup camera information
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.get_viewer_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = compute_view_params(
                camera_pos = p_ego,
                target_pos = p_trgt,
                up_vector  = np.array([0,0,1]),
            )
            self.set_viewer(
                azimuth   = cam_azimuth,
                distance  = cam_distance,
                elevation = cam_elevation,
                lookat    = cam_lookat,
                update    = True,
            )
        
        # Grab RGB and depth image
        rgb_img,depth_img = self.grab_rgbd_img() # get rgb and depth images

        # Resize depth image for reducing point clouds
        if rsz_rate_for_pcd is not None:
            h_rsz         = int(depth_img.shape[0]*rsz_rate_for_pcd)
            w_rsz         = int(depth_img.shape[1]*rsz_rate_for_pcd)
            depth_img_rsz = cv2.resize(depth_img,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
        else:
            depth_img_rsz = depth_img

        # Get PCD
        if fovy is None:
            if len(self.model.cam_fovy)==0: fovy = 45.0 # if cam is not defined, use 45 deg (default value)
            else: fovy = self.model.cam_fovy[0] # otherwise use the fovy of the first camera
        pcd,xyz_img,xyz_img_world = self.get_pcd_from_depth_img(depth_img_rsz,fovy=fovy) # [N x 3]

        # Resize rgb_image and depth_img (optional)
        if rsz_rate_for_img is not None:
            h = int(rgb_img.shape[0]*rsz_rate_for_img)
            w = int(rgb_img.shape[1]*rsz_rate_for_img)
            rgb_img   = cv2.resize(rgb_img,(w,h),interpolation=cv2.INTER_NEAREST)
            depth_img = cv2.resize(depth_img,(w,h),interpolation=cv2.INTER_NEAREST)

        # Restore view
        if restore_view:
            # Restore camera information
            self.set_viewer(
                azimuth   = viewer_azimuth,
                distance  = viewer_distance,
                elevation = viewer_elevation,
                lookat    = viewer_lookat,
                update    = True,
            )
        return rgb_img,depth_img,pcd,xyz_img,xyz_img_world
    
    def grab_image(self,rsz_rate=None,interpolation=cv2.INTER_NEAREST):
        """
        Capture the current rendered image from the viewer.
        
        Parameters:
            rsz_rate (float): Optional resize rate.
            interpolation: Interpolation method for resizing.
        
        Returns:
            np.array: The captured image.
        """
        img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)
        mujoco.mjr_readPixels(img, None,self.viewer.viewport,self.viewer.ctx)
        img = np.flipud(img) # flip image
        # Resize
        if rsz_rate is not None:
            h = int(img.shape[0]*rsz_rate)
            w = int(img.shape[1]*rsz_rate)
            img = cv2.resize(img,(w,h),interpolation=interpolation)
        # Backup
        if img.sum() > 0:
            self.grab_image_backup = img
        if img.sum() == 0: # use backup instead
            img = self.grab_image_backup
        return img.copy()
    
    def get_fixed_cam_rgb(self,cam_name):
        """
            Get RGB of fixed cam
        """
        # Parse camera information
        cam_idx  = self.cam_names.index(cam_name)
        cam      = self.cams[cam_idx]
        cam_fov  = self.cam_fovs[cam_idx]
        viewport = self.cam_viewports[cam_idx]
        # Update
        mujoco.mjv_updateScene(
            self.model,self.data,self.viewer.vopt,self.viewer.pert,
            cam,mujoco.mjtCatBit.mjCAT_ALL,self.viewer.scn)
        mujoco.mjr_render(viewport,self.viewer.scn,self.viewer.ctx)
        # Grab RGBD
        rgb = np.zeros((viewport.height,viewport.width,3),dtype=np.uint8)
        depth_raw = np.zeros((viewport.height,viewport.width),dtype=np.float32)
        mujoco.mjr_readPixels(rgb,depth_raw,viewport,self.viewer.ctx)
        rgb,depth_raw = np.flipud(rgb),np.flipud(depth_raw)
        return rgb
    
    def get_fixed_cam_rgbd_pcd(self,cam_name,downscale_pcd=0.1):
        """
        Capture RGB, depth images and point cloud data from a fixed camera.
        
        Parameters:
            cam_name (str): Name of the fixed camera.
            downscale_pcd (float): Downscaling factor for the point cloud.
        
        Returns:
            tuple: (rgb, depth, pcd, T_view) from the fixed camera.
        """
        # Parse camera information
        cam_idx  = self.cam_names.index(cam_name)
        cam      = self.cams[cam_idx]
        cam_fov  = self.cam_fovs[cam_idx]
        viewport = self.cam_viewports[cam_idx]
        # Update
        mujoco.mjv_updateScene(
            self.model,self.data,self.viewer.vopt,self.viewer.pert,
            cam,mujoco.mjtCatBit.mjCAT_ALL,self.viewer.scn)
        mujoco.mjr_render(viewport,self.viewer.scn,self.viewer.ctx)
        # Grab RGBD
        rgb = np.zeros((viewport.height,viewport.width,3),dtype=np.uint8)
        depth_raw = np.zeros((viewport.height,viewport.width),dtype=np.float32)
        mujoco.mjr_readPixels(rgb,depth_raw,viewport,self.viewer.ctx)
        rgb,depth_raw = np.flipud(rgb),np.flipud(depth_raw)
        # Rescale depth
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        depth = near/(1-depth_raw*(1-near/far))
        # Get PCD with resized depth image
        h_rsz = int(depth.shape[0]*downscale_pcd)
        w_rsz = int(depth.shape[1]*downscale_pcd)
        depth_rsz = cv2.resize(depth,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
        img_height,img_width = depth_rsz.shape[0],depth_rsz.shape[1]
        focal_scaling = 0.5*img_height/np.tan(cam_fov*np.pi/360)
        cam_matrix = np.array(((focal_scaling,0,img_width/2),
                               (0,focal_scaling,img_height/2),
                               (0,0,1))) # [3 x 3]
        xyz_img = meters2xyz(depth_rsz,cam_matrix) # [H x W x 3]
        xyz_transpose = np.transpose(xyz_img,(2,0,1)).reshape(3,-1) # [3 x N]
        xyzone_transpose = np.vstack((xyz_transpose,np.ones((1,xyz_transpose.shape[1])))) # [4 x N]
        # PCD to world coordinate
        T_view = self.get_T_cam(cam_name=cam_name)@pr2t(p=np.zeros(3),R=rpy2r(np.deg2rad([-45.,90.,45.])))
        xyzone_world_transpose = T_view @ xyzone_transpose
        xyz_world_transpose = xyzone_world_transpose[:3,:] # [3 x N]
        pcd = np.transpose(xyz_world_transpose,(1,0)) # [N x 3]
        # Return
        return rgb,depth,pcd,T_view
        
    def get_body_names(self,prefix='',excluding='world'):
        """
        Retrieve a list of body names starting with a given prefix and excluding specified names.
        
        Parameters:
            prefix (str): Prefix to match.
            excluding (str): Substring that, if present, excludes the body.
        
        Returns:
            list: Filtered body names.
        """
        body_names = [x for x in self.body_names if x is not None and x.startswith(prefix) and excluding not in x]
        return body_names
    
    def get_site_names(self,prefix='',excluding='world'):
        """
        Retrieve a list of site names starting with a given prefix and excluding specified names.
        
        Parameters:
            prefix (str): Prefix to match.
            excluding (str): Substring to exclude.
        
        Returns:
            list: Filtered site names.
        """
        site_names = [x for x in self.site_names if x is not None and x.startswith(prefix) and excluding not in x]
        return site_names
    
    def get_sensor_names(self,prefix='',excluding='world'):
        """
        Retrieve a list of sensor names starting with a given prefix and excluding specified names.
        
        Parameters:
            prefix (str): Prefix to match.
            excluding (str): Substring to exclude.
        
        Returns:
            list: Filtered sensor names.
        """
        sensor_names = [x for x in self.sensor_names if x is not None and x.startswith(prefix) and excluding not in x]
        return sensor_names
    
    def get_mesh_names(self,including='',excluding='collision'):
        """
        Retrieve a list of mesh names.
        """
        if excluding is None:
            mesh_names = [x for x in self.mesh_names if x is not None and including in x]    
        else:
            mesh_names = [x for x in self.mesh_names if x is not None and including in x and excluding not in x]
        return mesh_names
    
    def get_geom_idxs_from_body_name(self,body_name):
        """ 
            Get geometry indices from a body name
        """
        body_idx = self.body_names.index(body_name)
        geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx] 
        return geom_idxs

    def get_p_body(self,body_name):
        """
        Get the position of the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            np.array: The position of the body.
        """
        return self.data.body(body_name).xpos.copy()

    def get_R_body(self,body_name):
        """
        Get the rotation matrix of the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            np.array: The 3x3 rotation matrix.
        """
        return self.data.body(body_name).xmat.reshape([3,3]).copy()
    
    def get_T_body(self,body_name):
        """
        Get the full transformation matrix (pose) of the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            np.array: The 4x4 transformation matrix.
        """
        p_body = self.get_p_body(body_name=body_name)
        R_body = self.get_R_body(body_name=body_name)
        return pr2t(p_body,R_body)
    
    def get_pR_body(self,body_name):
        """
        Get both the position and rotation matrix of the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            tuple: (position, rotation matrix)
        """
        p = self.get_p_body(body_name)
        R = self.get_R_body(body_name)
        return p,R
    
    def get_p_joint(self,joint_name):
        """
        Get the position of the joint (via its associated body).
        
        Parameters:
            joint_name (str): Name of the joint.
        
        Returns:
            np.array: The joint position.
        """
        body_id = self.model.joint(joint_name).bodyid[0] # first body ID
        return self.get_p_body(self.body_names[body_id])

    def get_R_joint(self,joint_name):
        """
        Get the rotation matrix of the joint (via its associated body).
        
        Parameters:
            joint_name (str): Name of the joint.
        
        Returns:
            np.array: The joint rotation matrix.
        """
        body_id = self.model.joint(joint_name).bodyid[0] # first body ID
        return self.get_R_body(self.body_names[body_id])
    
    def get_pR_joint(self,joint_name):
        """
        Get both the position and rotation of the specified joint.
        
        Parameters:
            joint_name (str): Name of the joint.
        
        Returns:
            tuple: (position, rotation matrix)
        """
        p = self.get_p_joint(joint_name)
        R = self.get_R_joint(joint_name)
        return p,R
    
    def get_p_geom(self,geom_name):
        """
        Get the position of the specified geometry.
        
        Parameters:
            geom_name (str): Name of the geometry.
        
        Returns:
            np.array: The position of the geometry.
        """
        return self.data.geom(geom_name).xpos
    
    def get_R_geom(self,geom_name):
        """
        Get the rotation matrix of the specified geometry.
        
        Parameters:
            geom_name (str): Name of the geometry.
        
        Returns:
            np.array: The 3x3 rotation matrix.
        """
        return self.data.geom(geom_name).xmat.reshape((3,3))
    
    def get_pR_geom(self,geom_name):
        """
        Get both the position and rotation matrix of the specified geometry.
        
        Parameters:
            geom_name (str): Name of the geometry.
        
        Returns:
            tuple: (position, rotation matrix)
        """
        p = self.get_p_geom(geom_name)
        R = self.get_R_geom(geom_name)
        return p,R
    
    def get_site_name_of_sensor(self,sensor_name):
        """
        Retrieve the site name associated with the given sensor.
        
        Parameters:
            sensor_name (str): Name of the sensor.
        
        Returns:
            str: The corresponding site name.
        """
        sensor_id = self.model.sensor(sensor_name).id # get sensor ID
        sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
        sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
        return site_name
    
    def get_p_sensor(self,sensor_name):
        """
        Get the position of the sensor (via its associated site).
        
        Parameters:
            sensor_name (str): Name of the sensor.
        
        Returns:
            np.array: Sensor position.
        """
        sensor_id = self.model.sensor(sensor_name).id # get sensor ID
        sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
        sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
        p = self.data.site(site_name).xpos.copy() # get the position of the site
        return p
    
    def get_p_site(self,site_name):
        """
        Get the position of the specified site.
        
        Parameters:
            site_name (str): Name of the site.
        
        Returns:
            np.array: The position of the site.
        """
        return self.data.site(site_name).xpos.copy()
    
    def get_R_site(self,site_name):
        """
        Get the rotation matrix of the specified site.
        
        Parameters:
            site_name (str): Name of the site.
        
        Returns:
            np.array: The 3x3 rotation matrix.
        """
        return self.data.site(site_name).xmat.reshape(3,3).copy()
    
    def get_pR_site(self,site_name):
        """
        Get both the position and rotation matrix of the specified site.
        
        Parameters:
            site_name (str): Name of the site.
        
        Returns:
            tuple: (position, rotation matrix)
        """
        p_site = self.get_p_site(site_name)
        R_site = self.get_R_site(site_name)
        return p_site,R_site
    
    def get_R_sensor(self,sensor_name):
        """
        Get the rotation matrix of the specified sensor.
        
        Parameters:
            sensor_name (str): Name of the sensor.
        
        Returns:
            np.array: The sensor's rotation matrix.
        """
        sensor_id = self.model.sensor(sensor_name).id
        sensor_objtype = self.model.sensor_objtype[sensor_id]
        sensor_objid = self.model.sensor_objid[sensor_id]
        site_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid)
        R = self.data.site(site_name).xmat.reshape([3,3]).copy()
        return R
    
    def get_pR_sensor(self,sensor_name):
        """
        Get both the position and rotation of the specified sensor.
        
        Parameters:
            sensor_name (str): Name of the sensor.
        
        Returns:
            tuple: (position, rotation matrix)
        """
        p = self.get_p_sensor(sensor_name)
        R = self.get_R_sensor(sensor_name)
        return p,R
    
    def get_T_sensor(self,sensor_name):
        """
        Get the transformation matrix (pose) of the specified sensor.
        
        Parameters:
            sensor_name (str): Name of the sensor.
        
        Returns:
            np.array: The 4x4 transformation matrix.
        """
        p = self.get_p_sensor(sensor_name)
        R = self.get_R_sensor(sensor_name)
        return pr2t(p,R)
    
    def get_sensor_value(self,sensor_name):
        """
        Retrieve the current value of the specified sensor.
        
        Parameters:
            sensor_name (str): Name of the sensor.
        
        Returns:
            The sensor value.
        """
        data = self.data.sensor(sensor_name).data
        return data.copy()
    
    def get_sensor_values(self,sensor_names=None):
        """
        Retrieve the sensor values for multiple sensors.
        
        Parameters:
            sensor_names (list): List of sensor names. If None, returns all sensor values.
        
        Returns:
            np.array or list: The sensor values.
        """
        if sensor_names is None:
            sensor_names = self.sensor_names
        data = np.array([self.get_sensor_value(sensor_name) for sensor_name in self.sensor_names]).squeeze()
        if self.n_sensor == 1: return [data] # make it list
        else: return data.copy()
        
    def get_p_rf_list(self,sensor_names):
        """
        (Alias) Get the list of contact positions detected by range finder sensors.
        
        Parameters:
            sensor_names (list): List of sensor names.
        
        Returns:
            list: Contact positions.
        """
        return self.get_p_rf_obs_list(sensor_names)
        
    def get_p_rf_obs_list(self,sensor_names):
        """
        Get the contact positions between the range finder sensors and obstacles.
        
        Parameters:
            sensor_names (list): List of range finder sensor names.
        
        Returns:
            list: Observed contact positions.
        """
        p_rf_obs_list = []
        for sensor_name in sensor_names: # for all sensors
            rf_value      = self.get_sensor_value(sensor_name=sensor_name) # sensor value
            cutoff_val    = self.model.sensor(sensor_name).cutoff[0]
            if cutoff_val == 0: cutoff_val = np.inf
            site_name     = self.get_site_name_of_sensor(sensor_name=sensor_name) # site name
            p_site,R_site = self.get_pR_site(site_name=site_name) # site p and R
            if rf_value >= 0 and rf_value < cutoff_val:
                p_obs = p_site + rf_value*R_site[:,2] # z-axis if the ray direction
                p_rf_obs_list.append(p_obs) # append
        return p_rf_obs_list # list
    
    def get_p_cam(self,cam_name):
        """
        Get the position of the specified camera.
        
        Parameters:
            cam_name (str): Name of the camera.
        
        Returns:
            np.array: Camera position.
        """
        return self.data.cam(cam_name).xpos.copy()

    def get_R_cam(self,cam_name):
        """
        Get the rotation matrix of the specified camera.
        
        Parameters:
            cam_name (str): Name of the camera.
        
        Returns:
            np.array: 3x3 rotation matrix.
        """
        return self.data.cam(cam_name).xmat.reshape([3,3]).copy()
    
    def get_T_cam(self,cam_name):
        """
        Get the full transformation matrix (pose) of the specified camera.
        
        Parameters:
            cam_name (str): Name of the camera.
        
        Returns:
            np.array: 4x4 transformation matrix.
        """
        p_cam = self.get_p_cam(cam_name=cam_name)
        R_cam = self.get_R_cam(cam_name=cam_name)
        return pr2t(p_cam,R_cam)
    
    def plot_T(
            self,
            p           = np.array([0,0,0]),
            R           = np.eye(3),
            T           = None,
            plot_axis   = True,
            axis_len    = 1.0,
            axis_width  = 0.005,
            axis_rgba   = None,
            axis_alpha  = None,
            plot_sphere = False,
            sphere_r    = 0.05,
            sphere_rgba = [1,0,0,0.5],
            label       = None,
            print_xyz   = False,
        ):
        """
        Plot coordinate axes (and optionally a sphere and label) at the given pose.
        
        Parameters:
            p (np.array): Position.
            R (np.array): Rotation matrix. If T is provided, it overrides p and R.
            T (np.array): 4x4 transformation matrix.
            plot_axis (bool): Whether to plot the coordinate axes.
            axis_len (float): Length of each axis.
            axis_width (float): Thickness of the axes.
            axis_rgba (list): RGBA colors for the axes.
            plot_sphere (bool): Whether to plot a sphere marker at p.
            sphere_r (float): Radius of the sphere.
            sphere_rgba (list): RGBA color of the sphere.
            label (str): Optional text label.
            print_xyz (bool): Whether to print coordinate info.
        
        Returns:
            None
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
            
        if plot_axis:
            if axis_alpha is None: axis_alpha = 0.9
            if axis_rgba is None:
                rgba_x = [1.0,0.0,0.0,axis_alpha]
                rgba_y = [0.0,1.0,0.0,axis_alpha]
                rgba_z = [0.0,0.0,1.0,axis_alpha]
            else:
                rgba_x = axis_rgba
                rgba_y = axis_rgba
                rgba_z = axis_rgba
            R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
            p_x = p+R_x[:,2]*axis_len/2
            if print_xyz: axis_label = 'X-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_x,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_x,
                rgba  = rgba_x,
                label = axis_label,
            )
            R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
            p_y = p + R_y[:,2]*axis_len/2
            if print_xyz: axis_label = 'Y-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_y,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_y,
                rgba  = rgba_y,
                label = axis_label,
            )
            R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
            p_z = p + R_z[:,2]*axis_len/2
            if print_xyz: axis_label = 'Z-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_z,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_z,
                rgba  = rgba_z,
                label = axis_label,
            )

        if plot_sphere:
            self.viewer.add_marker(
                pos   = p,
                size  = [sphere_r,sphere_r,sphere_r],
                rgba  = sphere_rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = '')

        if label is not None:
            self.viewer.add_marker(
                pos   = p,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.01],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )

    def plot_sphere(self,p,r,rgba=[1,1,1,1],label=''):
        """
        Plot a sphere marker at the specified position.
        
        Parameters:
            p (np.array): Position (2D or 3D).
            r (float): Radius of the sphere.
            rgba (list): RGBA color.
            label (str): Optional label.
        
        Returns:
            None
        """
        p = np.asarray(p)
        if len(p) == 2: # only x and y are given (pad z=0)
            self.viewer.add_marker(
                pos   = np.append(p,[0]),
                size  = [r,r,r],
                rgba  = rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )
        elif len(p) == 3:
            self.viewer.add_marker(
                pos   = p,
                size  = [r,r,r],
                rgba  = rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )
        
    def plot_spheres(self,p_list,r,rgba=[1,1,1,1],label=''):
        """
        Plot multiple spheres at the positions given in p_list.
        
        Parameters:
            p_list (list of np.array): List of positions.
            r (float): Radius for each sphere.
            rgba (list): RGBA color.
            label (str): Optional label for each sphere.
        
        Returns:
            None
        """
        for p in p_list:
            self.plot_sphere(p=p,r=r,rgba=rgba,label=label)
                
    def plot_box(
            self,
            p     = np.array([0,0,0]),
            R     = np.eye(3),
            xlen  = 1.0,
            ylen  = 1.0,
            zlen  = 1.0,
            rgba  = [0.5,0.5,0.5,0.5],
            label = '',
        ):
        """
        Plot a box marker at the specified pose.
        
        Parameters:
            p (np.array): Position.
            R (np.array): Orientation matrix.
            xlen, ylen, zlen (float): Dimensions of the box.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_BOX,
            size  = [xlen/2,ylen/2,zlen/2],
            rgba  = rgba,
            label = label,
        )
    
    def plot_capsule(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        Plot a capsule marker.
        
        Parameters:
            p (np.array): Position.
            R (np.array): Orientation.
            r (float): Radius.
            h (float): Half-length of the capsule.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size  = [r,r,h],
            rgba  = rgba,
            label = label,
        )
        
    def plot_cylinder(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        Plot a cylinder marker.
        
        Parameters:
            p (np.array): Position.
            R (np.array): Orientation.
            r (float): Radius.
            h (float): Half-height.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,h],
            rgba  = rgba,
            label = label,
        )
    
    def plot_ellipsoid(self,p=np.array([0,0,0]),R=np.eye(3),rx=1.0,ry=1.0,rz=1.0,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        Plot an ellipsoid marker.
        
        Parameters:
            p (np.array): Position.
            R (np.array): Orientation.
            rx, ry, rz (float): Radii along x, y, z axes.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size  = [rx,ry,rz],
            rgba  = rgba,
            label = label,
        )
        
    def plot_arrow(self,p=np.array([0,0,0]),R=np.eye(3),r=1.0,h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        """
        Plot an arrow marker at the given pose.
        
        Parameters:
            p (np.array): Position.
            R (np.array): Orientation.
            r (float): Radius of the arrow shaft.
            h (float): Length of the arrow.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,h*2],
            rgba  = rgba,
            label = ''
        )
        
    def plot_line(self,p=np.array([0,0,0]),R=np.eye(3),h=1.0,rgba=[0.5,0.5,0.5,0.5]):
        """
        Plot a line marker.
        
        Parameters:
            p (np.array): Starting position.
            R (np.array): Orientation (direction).
            h (float): Length of the line.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = h,
            rgba  = rgba,
            label = ''
        )
        
    def plot_arrow_fr2to(self,p_fr,p_to,r=1.0,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        Plot an arrow from point p_fr to point p_to.
        
        Parameters:
            p_fr (np.array): Starting point.
            p_to (np.array): Ending point.
            r (float): Arrow shaft radius.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,np.linalg.norm(p_to-p_fr)*2],
            rgba  = rgba,
            label = label,
        )

    def plot_line_fr2to(self,p_fr,p_to,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        Plot a line connecting two points.
        
        Parameters:
            p_fr (np.array): Starting point.
            p_to (np.array): Ending point.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = np.linalg.norm(p_to-p_fr),
            rgba  = rgba,
            label = label,
        )
    
    def plot_cylinder_fr2to(self,p_fr,p_to,r=0.01,rgba=[0.5,0.5,0.5,0.5],label=''):
        """
        Plot a cylinder marker between two points.
        
        Parameters:
            p_fr (np.array): Starting point.
            p_to (np.array): Ending point.
            r (float): Cylinder radius.
            rgba (list): RGBA color.
        
        Returns:
            None
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        R_fr2to = get_rotation_matrix_from_two_points(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = (p_fr+p_to)/2,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,np.linalg.norm(p_to-p_fr)/2],
            rgba  = rgba,
            label = label,
        )
        
    def plot_traj(
            self,
            traj, # [L x 3] for (x,y,z) sequence or [L x 2] for (x,y) sequence
            rgba          = [1,0,0,1],
            plot_line     = False,
            plot_cylinder = True,
            plot_sphere   = False,
            cylinder_r    = 0.01,
            sphere_r      = 0.025,
        ):
        """
        Plot a trajectory given by a sequence of points.
        
        Parameters:
            traj (np.array): Array of shape [L x 3] or [L x 2] representing the trajectory.
            rgba (list): RGBA color for plotting.
            plot_line (bool): Whether to plot lines connecting points.
            plot_cylinder (bool): Whether to plot cylinders between points.
            plot_sphere (bool): Whether to plot spheres at the points.
            cylinder_r (float): Radius for the cylinder.
            sphere_r (float): Radius for the sphere.
        
        Returns:
            None
        """
        L = traj.shape[0]
        colors = None
        for idx in range(L-1):
            p_fr = traj[idx,:]
            p_to = traj[idx+1,:]
            if len(p_fr) == 2: p_fr = np.append(p_fr,[0])
            if len(p_to) == 2: p_to = np.append(p_to,[0])
            if plot_line:
                self.plot_line_fr2to(p_fr=p_fr,p_to=p_to,rgba=rgba)
            if plot_cylinder:
                self.plot_cylinder_fr2to(p_fr=p_fr,p_to=p_to,r=cylinder_r,rgba=rgba)
        if plot_sphere:
            for idx in range(L):
                p = traj[idx,:]
                self.plot_sphere(p=p,r=sphere_r,rgba=rgba)
        
    def plot_text(self,p,label=''):
        """
        Plot a text label at the specified position.
        
        Parameters:
            p (np.array): Position for the text.
            label (str): Text to display.
        
        Returns:
            None
        """
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            size  = [0.0001,0.0001,0.0001],
            rgba  = [1,1,1,0.01],
            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
            label = label,
        )

    def plot_time(
            self,
            loc = 'bottom left',
        ):
        """
        Overlay the current simulation tick, simulation time, and wall-clock time on the viewer.
        
        Parameters:
            loc (str): Location of the overlay.
        
        Returns:
            None
        """
        self.viewer.add_overlay(text1='tick',text2='%d'%(self.tick),loc=loc)
        self.viewer.add_overlay(text1='sim time',text2='%.2fsec'%(self.get_sim_time()),loc=loc)
        self.viewer.add_overlay(text1='wall time',text2='%.2fsec'%(self.get_wall_time()),loc=loc)
        
    def plot_sensor_T(
            self,
            sensor_name,
            plot_axis   = True,
            axis_len    = 0.1,
            axis_width  = 0.005,
            axis_rgba   = None,
            label       = None,
        ):
        """
        Plot the coordinate frame of a sensor.
        
        Parameters:
            sensor_name (str): Name of the sensor.
            plot_axis (bool): Whether to plot the axes.
            axis_len (float): Length of each axis.
            axis_width (float): Width of the axes.
            axis_rgba (list): RGBA color for the axes.
            label (str): Optional label.
        
        Returns:
            None
        """
        p_sensor,R_sensor = self.get_pR_sensor(sensor_name=sensor_name)
        self.plot_T(
            p_sensor,
            R_sensor,
            plot_axis   = plot_axis,
            axis_len    = axis_len,
            axis_width  = axis_width,
            axis_rgba   = axis_rgba,
            plot_sphere = False,
            label       = label,
        )
        
    def plot_sensors_T(
            self,
            sensor_names,
            plot_axis   = True,
            axis_len    = 0.1,
            axis_width  = 0.005,
            axis_rgba   = None,
            plot_name   = False,
        ):
        """
        Plot the coordinate frames of multiple sensors.
        
        Parameters:
            sensor_names (list): List of sensor names.
            plot_axis (bool): Whether to plot axes.
            axis_len (float): Axis length.
            axis_width (float): Axis width.
            axis_rgba (list): RGBA color.
            plot_name (bool): Whether to display sensor names.
        
        Returns:
            None
        """
        for sensor_idx,sensor_name in enumerate(sensor_names):
            if plot_name:
                label = '[%d] %s'%(sensor_idx,sensor_name)
            else:
                label = ''
            self.plot_sensor_T(
                sensor_name = sensor_name,
                plot_axis   = plot_axis,
                axis_len    = axis_len,
                axis_width  = axis_width,
                axis_rgba   = axis_rgba,
                label       = label,
             )
        
    def plot_sensors(
            self,
            loc = 'bottom right',
        ):
        """
        Overlay sensor values as text on the viewer.
        
        Parameters:
            loc (str): Location of the overlay.
        
        Returns:
            None
        """
        sensor_values = self.get_sensor_values() # print sensor values
        for sensor_idx,sensor_name in enumerate(self.sensor_names):
            self.viewer.add_overlay(
                text1 = '%s'%(sensor_name),
                text2 = '%.2f'%(sensor_values[sensor_idx]),
                loc   = loc,
            )

    def plot_body_T(
            self,
            body_name,
            plot_axis   = True,
            axis_len    = 0.1,
            axis_width  = 0.005,
            axis_rgba   = None,
            plot_sphere = False,
            sphere_r    = 0.05,
            sphere_rgba = [1,0,0,0.5],
            label       = None,
        ):
        """
        Plot the coordinate frame of a specified body.
        
        Parameters:
            body_name (str): Name of the body.
            plot_axis (bool): Whether to plot the axes.
            axis_len (float): Length of the axes.
            axis_width (float): Width of the axes.
            axis_rgba (list): RGBA color.
            plot_sphere (bool): Whether to plot a sphere marker.
            sphere_r (float): Sphere radius.
            sphere_rgba (list): Sphere color.
            label (str): Optional label.
        
        Returns:
            None
        """
        p,R = self.get_pR_body(body_name=body_name)
        self.plot_T(
            p,
            R,
            plot_axis   = plot_axis,
            axis_len    = axis_len,
            axis_width  = axis_width,
            axis_rgba   = axis_rgba,
            plot_sphere = plot_sphere,
            sphere_r    = sphere_r,
            sphere_rgba = sphere_rgba,
            label       = label,
        )

    def plot_body_sphere(
            self,
            body_name,
            r     = 0.05,
            rgba  = (1,0,0,0.5),
            label = None,
        ):
        """
        Plot the coordinate frame of a specified body.
        
        Parameters:
            body_name (str): Name of the body.
            plot_axis (bool): Whether to plot the axes.
            axis_len (float): Length of the axes.
            axis_width (float): Width of the axes.
            axis_rgba (list): RGBA color.
            plot_sphere (bool): Whether to plot a sphere marker.
            sphere_r (float): Sphere radius.
            sphere_rgba (list): Sphere color.
            label (str): Optional label.
        
        Returns:
            None
        """
        p,R = self.get_pR_body(body_name=body_name)
        self.plot_T(
            p,
            R,
            plot_axis   = False,
            axis_len    = None,
            axis_width  = None,
            axis_rgba   = None,
            plot_sphere = True,
            sphere_r    = r,
            sphere_rgba = rgba,
            label       = label,
        )
        
    def plot_joint_T(
            self,
            joint_name,
            plot_axis  = True,
            axis_len   = 1.0,
            axis_width = 0.01,
            axis_rgba  = None,
            label      = None,
        ):
        """
        Plot the coordinate frame of a specified joint.
        
        Parameters:
            joint_name (str): Name of the joint.
            plot_axis (bool): Whether to plot the axes.
            axis_len (float): Length of the axes.
            axis_width (float): Width of the axes.
            axis_rgba (list): RGBA color.
            label (str): Optional label.
        
        Returns:
            None
        """
        p,R = self.get_pR_joint(joint_name=joint_name)
        self.plot_T(
            p,
            R,
            plot_axis  = plot_axis,
            axis_len   = axis_len,
            axis_width = axis_width,
            axis_rgba  = axis_rgba,
            label      = label,
        )
        
    def plot_bodies_T(
            self,
            body_names            = None,
            body_names_to_exclude = [],
            body_names_to_exclude_including = [],
            plot_axis             = True,
            axis_len              = 0.05,
            axis_width            = 0.005,
            rate                  = 1.0,
            plot_name             = False,
        ):
        """
        Plot the coordinate frames of multiple bodies.
        
        Parameters:
            body_names (list): List of body names to plot (if None, plot all bodies).
            body_names_to_exclude (list): Body names to exclude.
            body_names_to_exclude_including (list): Bodies containing these substrings will be excluded.
            plot_axis (bool): Whether to plot axes.
            axis_len (float): Axis length.
            axis_width (float): Axis width.
            rate (float): Scaling factor.
            plot_name (bool): Whether to display body names.
        
        Returns:
            None
        """
        def should_exclude(x, exclude_list):
            for exclude in exclude_list:
                if exclude in x:
                    return True
            return False
        
        if body_names is None:
            body_names = self.body_names
            
        for body_idx,body_name in enumerate(body_names):
            if body_name in body_names_to_exclude: continue
            if body_name is None: continue
            
            if should_exclude(body_name,body_names_to_exclude_including): 
                # exclude body_name including ones in 'body_names_to_exclude_including'
                continue
            
            if plot_name:
                label = '[%d] %s'%(body_idx,body_name)
            else:
                label = ''
            self.plot_body_T(
                body_name  = body_name,
                plot_axis  = plot_axis,
                axis_len   = rate*axis_len,
                axis_width = rate*axis_width,
                label      = label,
            )
            
    def plot_links_between_bodies(
            self,
            parent_body_names_to_exclude = ['world'],
            body_names_to_exclude        = [],
            pbne                         = None,
            bne                          = None,
            r                            = 0.005,
            rgba                         = (0.0,0.0,0.0,0.5),
        ):
        """
        Plot visual links (e.g., cylinders) connecting parent and child bodies.
        
        Parameters:
            parent_body_names_to_exclude (list): Parent body names to exclude.
            body_names_to_exclude (list): Child body names to exclude.
            pbne, bne: Alternative exclusion lists.
            r (float): Radius of the linking cylinder.
            rgba (tuple): Color of the link.
        
        Returns:
            None
        """
        if pbne is not None: parent_body_names_to_exclude = pbne
        if bne is not None: body_names_to_exclude = bne
        for body_idx,body_name in enumerate(self.body_names):
            parent_body_name = self.parent_body_names[body_idx]
            if parent_body_name in parent_body_names_to_exclude: continue
            if body_name in body_names_to_exclude: continue
            if body_name is None: continue
            
            self.plot_cylinder_fr2to(
                p_fr = self.get_p_body(body_name=parent_body_name),
                p_to = self.get_p_body(body_name=body_name),
                r    = r,
                rgba = rgba,
            )

    def plot_joint_axis(
            self,
            axis_len    = 0.1,
            axis_r      = 0.01,
            joint_names = None,
            alpha       = 0.2,
            rate        = 1.0,
            print_name  = False,
        ):
        """
        Plot the axis of revolute joints.
        
        Parameters:
            axis_len (float): Length of the joint axis.
            axis_r (float): Radius of the axis marker.
            joint_names (list): List of joint names to plot.
            alpha (float): Transparency factor.
            rate (float): Scaling factor.
            print_name (bool): Whether to print joint names.
        
        Returns:
            None
        """
        rev_joint_idxs  = self.rev_joint_idxs
        rev_joint_names = self.rev_joint_names

        if joint_names is not None:
            idxs = get_idxs(self.rev_joint_names,joint_names)
            rev_joint_idxs_to_use  = rev_joint_idxs[idxs]
            rev_joint_names_to_use = [rev_joint_names[i] for i in idxs]
        else:
            rev_joint_idxs_to_use  = rev_joint_idxs
            rev_joint_names_to_use = rev_joint_names

        for rev_joint_idx,rev_joint_name in zip(rev_joint_idxs_to_use,rev_joint_names_to_use):
            axis_joint      = self.model.jnt_axis[rev_joint_idx]
            p_joint,R_joint = self.get_pR_joint(joint_name=rev_joint_name)
            axis_world      = R_joint@axis_joint
            axis_rgba       = np.append(np.eye(3)[:,np.argmax(np.abs(axis_joint))],alpha)
            self.plot_arrow_fr2to(
                p_fr = p_joint,
                p_to = p_joint+rate*axis_len*axis_world,
                r    = rate*axis_r,
                rgba = axis_rgba
            )
            if print_name:
                self.plot_text(p=p_joint,label=rev_joint_name)
                
    def get_contact_body_names(self):
        """
        Retrieve the names of the bodies involved in each contact.
        
        Returns:
            list: List of pairs [body1, body2] for each contact.
        """
        contact_body_names = []
        for c_idx in range(self.data.ncon):
            contact = self.data.contact[c_idx]
            contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
            contact_body2 = self.body_names[self.model.geom_bodyid[contact.geom2]]
            contact_body_names.append([contact_body1,contact_body2])
        return contact_body_names
    
    def get_contact_info(self,must_include_prefix=None,must_exclude_prefix=None):
        """
        Retrieve detailed contact information including positions, forces, and involved geometries and bodies.
        
        Parameters:
            must_include_prefix (str): Only include contacts where one of the geometry names starts with this prefix.
            must_exclude_prefix (str): Exclude contacts where geometry names start with this prefix.
        
        Returns:
            tuple: (p_contacts, f_contacts, geom1s, geom2s, body1s, body2s)
        """
        p_contacts = []
        f_contacts = []
        geom1s = []
        geom2s = []
        body1s = []
        body2s = []
        for c_idx in range(self.data.ncon):
            contact   = self.data.contact[c_idx]
            # Contact position and frame orientation
            p_contact = contact.pos # contact position
            R_frame   = contact.frame.reshape(( 3,3))
            # Contact force
            f_contact_local = np.zeros(6,dtype=np.float64)
            mujoco.mj_contactForce(self.model,self.data,0,f_contact_local)
            f_contact = R_frame @ f_contact_local[:3] # in the global coordinate
            # Contacting geoms
            contact_geom1 = self.geom_names[contact.geom1]
            contact_geom2 = self.geom_names[contact.geom2]
            contact_body1 = self.body_names[self.model.geom_bodyid[contact.geom1]]
            contact_body2 = self.body_names[self.model.geom_bodyid[contact.geom2]]
            # Append
            if must_include_prefix is not None:
                if (contact_geom1[:len(must_include_prefix)] == must_include_prefix) or \
                (contact_geom2[:len(must_include_prefix)] == must_include_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            elif must_exclude_prefix is not None:
                if (contact_geom1[:len(must_exclude_prefix)] != must_exclude_prefix) and \
                    (contact_geom2[:len(must_exclude_prefix)] != must_exclude_prefix):
                    p_contacts.append(p_contact)
                    f_contacts.append(f_contact)
                    geom1s.append(contact_geom1)
                    geom2s.append(contact_geom2)
                    body1s.append(contact_body1)
                    body2s.append(contact_body2)
            else:
                p_contacts.append(p_contact)
                f_contacts.append(f_contact)
                geom1s.append(contact_geom1)
                geom2s.append(contact_geom2)
                body1s.append(contact_body1)
                body2s.append(contact_body2)
        return p_contacts,f_contacts,geom1s,geom2s,body1s,body2s

    def print_contact_info(self,must_include_prefix=None):
        """
        Print contact information for contacts matching specified criteria.
        
        Parameters:
            must_include_prefix (str): Filter to only include contacts with geometries starting with this prefix.
        
        Returns:
            None
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            print ("Tick:[%d] Body contact:[%s]-[%s]"%(self.tick,body1,body2))

    def plot_arrow_contact(self,p,uv,r_arrow=0.03,h_arrow=0.3,rgba=[1,0,0,1],label=''):
        """
        Plot an arrow representing a contact force at a given contact point.
        
        Parameters:
            p (np.array): Contact position.
            uv (np.array): Unit vector indicating force direction.
            r_arrow (float): Radius of the arrow.
            h_arrow (float): Length of the arrow.
            rgba (list): RGBA color.
            label (str): Optional label.
        
        Returns:
            None
        """
        p_a = np.copy(np.array([0,0,1]))
        p_b = np.copy(uv)
        p_a_norm = np.linalg.norm(p_a)
        p_b_norm = np.linalg.norm(p_b)
        if p_a_norm > 1e-9: p_a = p_a/p_a_norm
        if p_b_norm > 1e-9: p_b = p_b/p_b_norm
        v = np.cross(p_a,p_b)
        S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        if np.linalg.norm(v) == 0:
            R = np.eye(3,3)
        else:
            R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r_arrow,r_arrow,h_arrow],
            rgba  = rgba,
            label = label
        )

    def plot_joints(
            self,
            joint_names      = None,
            plot_axis        = True,
            axis_len         = 0.1,
            axis_width       = 0.01,
            axis_rgba        = None,
            plot_joint_names = False,
        ):
        """
        Plot the coordinate frames for multiple joints.
        
        Parameters:
            joint_names (list): List of joint names. If None, all joints are plotted.
            plot_axis (bool): Whether to display axes.
            axis_len (float): Length of axes.
            axis_width (float): Width of axes.
            axis_rgba (list): RGBA color.
            plot_joint_names (bool): Whether to print joint names.
        
        Returns:
            None
        """
        if joint_names is None:
            joint_names = self.joint_names
        for joint_name in joint_names:
            if joint_name is not None:
                if plot_joint_names:
                    label = joint_name
                else:
                    label = None
                self.plot_joint_T(
                    joint_name,
                    plot_axis  = plot_axis,
                    axis_len   = axis_len,
                    axis_width = axis_width,
                    axis_rgba  = axis_rgba,
                    label      = label,
                )

    def plot_contact_info(
            self,
            must_include_prefix = None,
            plot_arrow          = True,
            r_arrow             = 0.005,
            h_arrow             = 0.1,
            rate                = 1.0,
            plot_sphere         = False,
            r_sphere            = 0.02,
            rgba_contact        = [1,0,0,1],
            print_contact_body  = False,
            print_contact_geom  = False,
            verbose             = False
        ):
        """
        Visualize contact forces and optionally display contact labels.
        
        Parameters:
            must_include_prefix (str): Filter for contacts.
            plot_arrow (bool): Whether to plot arrows for contact forces.
            r_arrow (float): Arrow radius.
            h_arrow (float): Arrow length.
            rate (float): Scaling factor.
            plot_sphere (bool): Whether to plot a sphere at contact points.
            r_sphere (float): Sphere radius.
            rgba_contact (list): RGBA color for contact markers.
            print_contact_body (bool): Whether to display contacting body names.
            print_contact_geom (bool): Whether to display contacting geometry names.
            verbose (bool): If True, also print contact info to console.
        
        Returns:
            None
        """
        # Get contact information
        p_contacts,f_contacts,geom1s,geom2s,body1s,body2s = self.get_contact_info(
            must_include_prefix=must_include_prefix)
        # Render contact informations
        for (p_contact,f_contact,geom1,geom2,body1,body2) in zip(p_contacts,f_contacts,geom1s,geom2s,body1s,body2s):
            f_norm = np.linalg.norm(f_contact)
            f_uv   = f_contact / (f_norm+1e-8)
            # h_arrow = 0.3 # f_norm*0.05
            if plot_arrow:
                self.plot_arrow_contact(
                    p       = p_contact,
                    uv      = f_uv,
                    r_arrow = rate*r_arrow,
                    h_arrow = rate*h_arrow,
                    rgba    = rgba_contact,
                    label   = '',
                )
                self.plot_arrow_contact(
                    p       = p_contact,
                    uv      = -f_uv,
                    r_arrow = rate*r_arrow,
                    h_arrow = rate*h_arrow,
                    rgba    = rgba_contact,
                    label   = '',
                )
            if plot_sphere: 
                # contact_label = '[%s]-[%s]'%(body1,body2)
                contact_label = ''
                self.plot_sphere(p=p_contact,r=r_sphere,rgba=rgba_contact,label=contact_label)
            if print_contact_body:
                label = '[%s]-[%s]'%(body1,body2)
            elif print_contact_geom:
                label = '[%s]-[%s]'%(geom1,geom2)
            else:
                label = '' 
        # Print
        if verbose:
            self.print_contact_info(must_include_prefix=must_include_prefix)
            
    def plot_xy_heading(
            self,
            xy,
            heading,
            r             = 0.01,
            arrow_len     = 0.1,
            rgba          = (1,0,0,1),
            plot_sphere   = False,
            plot_arrow    = True,
        ):
        """
        Plot a 2D point along with an arrow indicating its heading.
        
        Parameters:
            xy (np.array): (x, y) position.
            heading (float): Heading angle in radians.
            r (float): Radius for the point marker.
            arrow_len (float): Length of the heading arrow.
            rgba (tuple): RGBA color.
            plot_sphere (bool): Whether to plot a sphere at the point.
            plot_arrow (bool): Whether to draw the heading arrow.
        
        Returns:
            None
        """
        dir_vec = np.array([np.cos(heading),np.sin(heading)])
        if plot_sphere:
            self.plot_sphere(p=np.append(xy,[0]),r=r,rgba=rgba)
        if plot_arrow:
            self.plot_arrow_fr2to(
                p_fr = np.append(xy,[0]),
                p_to = np.append(xy+arrow_len*dir_vec,[0]),
                r    = r,
                rgba = rgba,
            )
                    
    def plot_xy_heading_traj(
            self,
            xy_traj,
            heading_traj,
            r             = 0.01,
            arrow_len     = 0.1,
            rgba          = None,
            cmap_name     = 'gist_rainbow',
            alpha         = 0.5,
            plot_sphere   = False,
            plot_arrow    = True,
            plot_cylinder = False,
        ):
        """
        Plot a trajectory in the XY plane with associated heading arrows.
        
        Parameters:
            xy_traj (np.array): Sequence of (x, y) positions.
            heading_traj (np.array): Sequence of heading angles.
            r (float): Marker radius.
            arrow_len (float): Length of the heading arrow.
            rgba (list): RGBA color; if None, use a colormap.
            cmap_name (str): Name of the colormap to use.
            alpha (float): Transparency for the colormap.
            plot_sphere (bool): Whether to plot spheres at points.
            plot_arrow (bool): Whether to draw arrows for headings.
            plot_cylinder (bool): Whether to connect points with cylinders.
        
        Returns:
            None
        """
        L = len(xy_traj)
        colors = get_colors(n_color=L,cmap_name=cmap_name,alpha=alpha)
        for idx in range(L):
            xy_i,heading_i = xy_traj[idx],heading_traj[idx]
            if rgba is None:
                rgba = colors[idx]
            dir_vec_i = np.array([np.cos(heading_i),np.sin(heading_i)])
            if plot_sphere:
                self.plot_sphere(p=np.append(xy_i,[0]),r=r,rgba=rgba)
            if plot_arrow:
                self.plot_arrow_fr2to(
                    p_fr = np.append(xy_i,[0]),
                    p_to = np.append(xy_i+arrow_len*dir_vec_i,[0]),
                    r    = r,
                    rgba = rgba,
                )
            if plot_cylinder:
                if idx > 1:
                    xy_prev = xy_traj[idx-1]
                    self.plot_cylinder_fr2to(
                        p_fr = np.append(xy_prev,[0]),
                        p_to = np.append(xy_i,[0]),
                        r    = r,
                        rgba = rgba,
                    )
            
    def get_idxs_fwd(self,joint_names):
        """
        Get the indices of joints used for forward kinematics based on joint names.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            list: Indices corresponding to the joints.

        Example:
            env.forward(q=q,joint_idxs=idxs_fwd) # <= HERE
        """
        return [self.model.joint(jname).qposadr[0] for jname in joint_names]
    
    def get_idxs_jac(self,joint_names):
        """ 
        Get the indices of joints for Jacobian calculation based on joint names.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            list: Indices corresponding to the joints.
        """
        return [self.model.joint(jname).dofadr[0] for jname in joint_names]
    
    def get_idxs_step(self,joint_names):
        """
        Get the indices used for applying control during simulation steps based on joint names.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            list: Control indices.
        """
        return [self.ctrl_qpos_names.index(jname) for jname in joint_names]
    
    def get_qpos(self):
        """
        Retrieve the current joint positions.
        
        Returns:
            np.array: The joint positions.
        """
        return self.data.qpos.copy() # [n_qpos]
    
    def get_qvel(self):
        """
        Retrieve the current joint velocities.
        
        Returns:
            np.array: The joint velocities.
        """
        return self.data.qvel.copy() # [n_qvel]
    
    def get_qacc(self):
        """
        Retrieve the current joint accelerations.
        
        Returns:
            np.array: The joint accelerations.
        """
        return self.data.qacc.copy() # [n_qacc]

    def get_qpos_joint(self,joint_name):
        """
        Get the position for a specific joint.
        
        Parameters:
            joint_name (str): Name of the joint.
        
        Returns:
            np.array: The joint position.
        """
        addr = self.model.joint(joint_name).qposadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        qpos = self.data.qpos[addr:addr+L]
        return qpos
    
    def get_qvel_joint(self,joint_name):
        """
        Get the velocity for a specific joint.
        
        Parameters:
            joint_name (str): Name of the joint.
        
        Returns:
            np.array: The joint velocity.
        """
        addr = self.model.joint(joint_name).dofadr[0]
        L = len(self.model.joint(joint_name).qpos0)
        if L > 1: L = 6
        qvel = self.data.qvel[addr:addr+L]
        return qvel
    
    def get_qpos_joints(self,joint_names):
        """
        Get the positions for multiple joints.
        
        Parameters:
            joint_names (list): List of joint names.
        
        Returns:
            np.array: Joint positions.
        """
        return np.array([self.get_qpos_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def get_qvel_joints(self,joint_names):
        """
        Get the velocity for a specific joint.
        
        Parameters:
            joint_name (str): Name of the joint.
        
        Returns:
            np.array: The joint velocity.
        """
        return np.array([self.get_qvel_joint(joint_name) for joint_name in joint_names]).squeeze()
    
    def get_q_couple(
        self,
        q_raw,
        coupled_joint_idxs_list    = None,
        coupled_joint_names_list   = None,
        coupled_joint_weights_list = None,
        ):
        """
        Compute coupled joint positions based on raw joint positions and coupling definitions.
        
        Parameters:
            q_raw (np.array): Raw joint position vector.
            coupled_joint_idxs_list (list): List of lists of joint indices for each coupling group.
            coupled_joint_names_list (list): Alternative specification using joint names.
            coupled_joint_weights_list (list): List of weight lists for each coupling group.
        
        Returns:
            np.array: Modified joint position vector with coupling applied.
        
        Usage?
            Coupled joint positions
            Example:
            # Apply joint positions coupling
            coupled_joint_idxs_list = [
                [22,23],[24,25,26],[27,28,29],[30,31,32],[33,34,35],
                [45,46],[47,48,49],[50,51,52],[53,54,55],[56,57,58]]
            coupled_joint_weights_list = [
                [1,1],[1,3,2],[1,3,2],[1,3,2],[1,3,2],
                [1,1],[1,3,2],[1,3,2],[1,3,2],[1,3,2]]
            q_couple = env.get_q_couple(
                q_raw=env.data.qpos,
                coupled_joint_idxs_list=coupled_joint_idxs_list,
                coupled_joint_weights_list=coupled_joint_weights_list)
        """
        q_couple = q_raw.copy()
        if coupled_joint_idxs_list is not None:
            for i in range(len(coupled_joint_idxs_list)): # for each couple
                coupled_joint_idxs    = coupled_joint_idxs_list[i]
                coupled_joint_weights = coupled_joint_weights_list[i]
                joint_sum = 0
                for j in range(len(coupled_joint_idxs)):
                    joint_sum += q_raw[coupled_joint_idxs[j]]
                joint_sum /= np.sum(coupled_joint_weights)
                for k in range(len(coupled_joint_idxs)):
                    q_couple[coupled_joint_idxs[k]] = joint_sum*coupled_joint_weights[k] # distribute coupled joint positions
        if coupled_joint_names_list is not None:
            for i in range(len(coupled_joint_names_list)): # for each couple
                coupled_joint_names   = coupled_joint_names_list[i]
                coupled_joint_idxs    = get_idxs(self.joint_names,coupled_joint_names)
                coupled_joint_weights = coupled_joint_weights_list[i]
                joint_sum = 0
                for j in range(len(coupled_joint_idxs)):
                    joint_sum += q_raw[coupled_joint_idxs[j]]
                joint_sum /= np.sum(coupled_joint_weights)
                for k in range(len(coupled_joint_idxs)):
                    q_couple[coupled_joint_idxs[k]] = joint_sum*coupled_joint_weights[k] # distribute coupled joint positions
        return q_couple
    
    def get_ctrl(self,ctrl_names):
        """
        Retrieve control values for the specified actuators.
        
        Parameters:
            ctrl_names (list): List of control names.
        
        Returns:
            np.array: Control values.
        """
        idxs = get_idxs(self.ctrl_names,ctrl_names)
        return np.array([self.data.ctrl[idx] for idx in idxs]).squeeze()
    
        
    def set_qpos_joints(self,joint_names,qpos):
        """
        Set the joint positions for the specified joints and update forward kinematics.
        
        Parameters:
            joint_names (list): Names of the joints.
            qpos (np.array): Joint positions.
        
        Returns:
            None
        """
        joint_idxs = self.get_idxs_fwd(joint_names)
        self.data.qpos[joint_idxs] = qpos
        mujoco.mj_forward(self.model,self.data)
    
    def set_ctrl(self,ctrl_names,ctrl,nstep=1):
        """
        Set control inputs for the specified actuators and perform simulation steps.
        
        Parameters:
            ctrl_names (list): Names of the controls.
            ctrl (np.array): Control values.
            nstep (int): Number of simulation steps to execute.
        
        Returns:
            None
        """
        ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)
        self.data.ctrl[ctrl_idxs] = ctrl
        mujoco.mj_step(self.model,self.data,nstep=nstep)
        
    def viewer_pause(self):
        """
        Pause the viewer rendering loop.
        
        Returns:
            None
        """
        self.viewer._paused = True
        
    def viewer_resume(self):
        """
        Resume the viewer rendering loop.
        
        Returns:
            None
        """
        self.viewer._paused = False
    
    def get_viewer_mouse_xy(self):
        """
        Get the current mouse (x, y) coordinates from the viewer.
        
        Returns:
            np.array: Mouse coordinates.
        """
        viewer_mouse_xy = np.array([self.viewer._last_mouse_x,self.viewer._last_mouse_y])
        return viewer_mouse_xy
    
    def get_xyz_left_double_click(self,verbose=False,fovy=45):
        """ 
            Get xyz location of double click
            :return self.xyz_left_double_click,flag_click:
        """
        flag_click = False
        if self.viewer._left_double_click_pressed: # left double click
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd(fovy=fovy)
            self.xyz_left_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._left_double_click_pressed = False
            flag_click = True
            if verbose:
                print ("left double click:(%.3f,%.3f,%.3f)"%
                       (self.xyz_left_double_click[0],self.xyz_left_double_click[1],self.xyz_left_double_click[2]))
        return self.xyz_left_double_click,flag_click
    
    def is_left_double_clicked(self):
        """
        Check if a left double-click event has occurred.
        
        Returns:
            bool: True if detected, False otherwise.
        """
        if self.viewer._left_double_click_pressed: # left double click
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd()
            self.xyz_left_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._left_double_click_pressed = False # toggle flag
            return True 
        else:
            return False
    
    def get_xyz_right_double_click(self,verbose=False,fovy=45):
        """
        Retrieve the 3D world coordinates corresponding to a right double-click event.
        
        Parameters:
            verbose (bool): If True, print the clicked coordinates.
            fovy (float): Field of view used for projection.
        
        Returns:
            tuple: (xyz, flag_click)
        """
        flag_click = False
        if self.viewer._right_double_click_pressed: # right double click
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd(fovy=fovy)
            self.xyz_right_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._right_double_click_pressed = False
            flag_click = True
            if verbose:
                print ("right double click:(%.3f,%.3f,%.3f)"%
                       (self.xyz_right_double_click[0],self.xyz_right_double_click[1],self.xyz_right_double_click[2]))
        return self.xyz_right_double_click,flag_click
    
    def is_right_double_clicked(self):
        """
        Check if a right double-click event has occurred.
        
        Returns:
            bool: True if detected, False otherwise.
        """
        if self.viewer._right_double_click_pressed: # right double click
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            _,_,_,_,xyz_img_world = self.get_egocentric_rgbd_pcd()
            self.xyz_right_double_click = xyz_img_world[int(viewer_mouse_xy[1]),int(viewer_mouse_xy[0])]
            self.viewer._right_double_click_pressed = False # toggle flag
            return True 
        else:
            return False
        
    def get_body_name_closest(self,xyz,body_names=None,verbose=False):
        """
        Determine which body is closest to the given 3D point.
        
        Parameters:
            xyz (np.array): The query 3D point.
            body_names (list): List of body names to consider (if None, all bodies are considered).
            verbose (bool): If True, print the selected body.
        
        Returns:
            tuple: (body_name_closest, p_body_closest)
        """
        if body_names is None:
            body_names = self.body_names
        dists = np.zeros(len(body_names))
        p_body_list = []
        for body_idx,body_name in enumerate(body_names):
            p_body = self.get_p_body(body_name=body_name)
            dist = np.linalg.norm(p_body-xyz)
            dists[body_idx] = dist # append
            p_body_list.append(p_body) # append
        idx_min = np.argmin(dists)
        body_name_closest = body_names[idx_min]
        p_body_closest = p_body_list[idx_min]
        if verbose:
            print ("[%s] selected"%(body_name_closest))
        return body_name_closest,p_body_closest
    
    # Inverse kinematics
    def get_J_body(self,body_name):
        """
        Compute the Jacobian matrices (position and rotation) for the specified body.
        
        Parameters:
            body_name (str): Name of the body.
        
        Returns:
            tuple: (J_p, J_R, J_full) where J_full is the stacked Jacobian.
        """
        J_p = np.zeros((3,self.n_dof)) # nv: nDoF
        J_R = np.zeros((3,self.n_dof))
        mujoco.mj_jacBody(self.model,self.data,J_p,J_R,self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_J_geom(self,geom_name):
        """
        Compute the Jacobian matrices for the specified geometry.
        
        Parameters:
            geom_name (str): Name of the geometry.
        
        Returns:
            tuple: (J_p, J_R, J_full)
        """
        J_p = np.zeros((3,self.n_dof)) # nv: nDoF
        J_R = np.zeros((3,self.n_dof))
        mujoco.mj_jacGeom(self.model,self.data,J_p,J_R,self.data.geom(geom_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_ik_ingredients(
            self,
            body_name = None,
            geom_name = None,
            p_trgt    = None,
            R_trgt    = None,
            IK_P      = True,
            IK_R      = True,
        ):
        """
        Compute the Jacobian and error vector needed for inverse kinematics.
        
        Parameters:
            body_name (str): Name of the body (if provided).
            geom_name (str): Name of the geometry (if provided).
            p_trgt (np.array): Target position.
            R_trgt (np.array): Target rotation matrix.
            IK_P (bool): Whether to include position error.
            IK_R (bool): Whether to include orientation error.
        
        Returns:
            tuple: (J, err) where J is the Jacobian and err is the error vector.
        """

        if p_trgt is None: IK_P = False
        if R_trgt is None: IK_R = False

        if body_name is not None:
            J_p,J_R,J_full = self.get_J_body(body_name=body_name)
            p_curr,R_curr = self.get_pR_body(body_name=body_name)
        if geom_name is not None:
            J_p,J_R,J_full = self.get_J_geom(geom_name=geom_name)
            p_curr,R_curr = self.get_pR_geom(geom_name=geom_name)
        if (body_name is not None) and (geom_name is not None):
            print ("[get_ik_ingredients] body_name:[%s] geom_name:[%s] are both not None!"%(body_name,geom_name))
        if (IK_P and IK_R):
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_full
            err   = np.concatenate((p_err,w_err))
        elif (IK_P and not IK_R):
            p_err = (p_trgt-p_curr)
            J     = J_p
            err   = p_err
        elif (not IK_P and IK_R):
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J     = J_R
            err   = w_err
        else:
            J   = None
            err = None
        return J,err
    
    def damped_ls(self,J,err,eps=1e-6,stepsize=1.0,th=5*np.pi/180.0):
        """
        Solve the inverse kinematics using the damped least squares method.
        
        Parameters:
            J (np.array): Jacobian matrix.
            err (np.array): Error vector.
            eps (float): Damping factor.
            stepsize (float): Step size multiplier.
            th (float): Threshold for scaling the result.
        
        Returns:
            np.array: The computed joint increments (dq).
        """
        dq = stepsize*np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        dq = trim_scale(x=dq,th=th)
        return dq
    
    def check_key_pressed(self,char=None):
        """
        High-level function to check if a key has been pressed.
        
        Parameters:
            char (str): Single character to check.
        Returns:
            bool: True if the key was pressed, False otherwise.
        """
        if self.viewer._is_key_pressed:
            if self.get_key_pressed() == char:
                self.viewer._is_key_pressed = False
                return True
            else:
                return False
        else:
            return False
        
    def get_key_pressed(self):
        """
        Retrieve the last key that was pressed.
        
        Returns:
            str: The key that was pressed.
        """
        return self.viewer._key_pressed
    
    def open_interactive_viewer(self):
        """
        Launch an interactive viewer for the simulation.
        
        Returns:
            None
        """
        from mujoco import viewer
        viewer.launch(self.model)
        
    def compensate_gravity(self,root_body_names):
        """
            Gravity compensation
        """
        qfrc_applied = self.data.qfrc_applied
        qfrc_applied[:] = 0.0  # Don't accumulate from previous calls.
        jac = np.empty((3,self.model.nv))
        for root_body_name in root_body_names:
            subtree_id = self.model.body(root_body_name).id
            total_mass = self.model.body_subtreemass[subtree_id]
            mujoco.mj_jacSubtreeCom(self.model,self.data,jac,subtree_id)
            qfrc_applied[:] -= self.model.opt.gravity * total_mass @ jac
            
    def set_rangefinder_rgba(self,rgba=(1,1,0,0.1)):
        """
        Set the RGBA color for the rangefinder visualization.
        
        Parameters:
            rgba (tuple): Color in RGBA format.
        
        Returns:
            None
        """
        self.model.vis.rgba.rangefinder = np.array(rgba,dtype=np.float32)
        
    def tic(self):
        """ 
        Start a timer for performance measurement.
        
        Returns:
            None
        """
        self.tt.tic()
        
    def toc(self):
        """
        Return the elapsed time since the last tic() call.
        
        Returns:
            float: Elapsed time in seconds.
        """
        return self.tt.toc()
        
    def sync_sim_wall_time(self):
        """
        Synchronize the simulation time with the wall time.
        
        Returns:
            None
        """
        time_diff = self.get_sim_time() - self.get_wall_time()
        if time_diff > 0: time.sleep(time_diff)

    def get_key_pressed_list(self):
        """
        Get the list of keys that have been pressed.
        
        Returns:
            list: List of pressed keys.
        """
        return list(self.viewer._key_pressed_set)
    
    def get_key_repeated_list(self):
        """
        Get the list of keys that have been repeatedly pressed.
        
        Returns:
            list: List of repeatedly pressed keys.
        """
        return list(self.viewer._key_repeated_set)
    
    def pop_key_pressed_list(self,key=None):
        """
        Pop the last key from the pressed keys list.
        
        Returns:
            str: The last pressed key.
        """
        if key is not None:
            self.viewer._key_pressed_set.discard(key)

    def is_key_pressed_once(self,key=None,key_list=None):
        """
        Check if a specific key has been pressed once.
        
        Parameters:
            key (str): Key to check.
        
        Returns:
            bool: True if the key was pressed, False otherwise.
        """
        if key is not None:
            if key in self.get_key_pressed_list():
                self.pop_key_pressed_list(key=key)
                return True
            else:
                return False
        elif key_list is not None:
            for key in key_list:
                if key in self.get_key_pressed_list():
                    self.pop_key_pressed_list(key=key)
                    return True
            return False
        else:
            return False
        
    def is_key_pressed_repeat(self,key=None,key_list=None):
        """
        Check if specific key(s) have been pressed continuously.
        
        Parameters:
            key (str, optional): Single key to check.
            key_list (list of str, optional): List of keys to check.
                If both key and key_list are provided, only 'key' is used.

        Returns:
            bool: True if the key (or any key in key_list) is pressed, False otherwise.
        """
        if key is not None:
            return key in self.get_key_pressed_list()+self.get_key_repeated_list()
        elif key_list is not None:
            for key in key_list:
                if key in self.get_key_pressed_list()+self.get_key_repeated_list():
                    return True
            return False
        else:
            return False
    