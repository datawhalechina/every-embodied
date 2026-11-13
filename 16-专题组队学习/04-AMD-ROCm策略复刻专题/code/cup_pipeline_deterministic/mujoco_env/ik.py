import sys
import numpy as np
from .utils import (
    get_colors,
    get_idxs,
)

# Inverse kinematics helper
def init_ik_info():
    """
        Initialize IK information
        Usage:
        ik_info = init_ik_info()
        ...
        add_ik_info(ik_info,body_name='BODY_NAME',p_trgt=P_TRGT,R_trgt=R_TRGT)
        ...
        for ik_tick in range(max_ik_tick):
            dq,ik_err_stack = get_dq_from_ik_info(
                env = env,
                ik_info = ik_info,
                stepsize = 1,
                eps = 1e-2,
                th = np.radians(10.0),
                joint_idxs_jac = joint_idxs_jac,
            )
            qpos = env.get_qpos()
            mujoco.mj_integratePos(env.model,qpos,dq,1)
            env.forward(q=qpos)
            if np.linalg.norm(ik_err_stack) < 0.05: break
    """
    ik_info = {
        'body_names':[],
        'geom_names':[],
        'p_trgts':[],
        'R_trgts':[],
        'n_trgt':0,
    }
    return ik_info

def add_ik_info(
        ik_info,
        body_name = None,
        geom_name = None,
        p_trgt    = None,
        R_trgt    = None,
    ):
    """ 
        Add IK information
    """
    ik_info['body_names'].append(body_name)
    ik_info['geom_names'].append(geom_name)
    ik_info['p_trgts'].append(p_trgt)
    ik_info['R_trgts'].append(R_trgt)
    ik_info['n_trgt'] = ik_info['n_trgt'] + 1
    
def get_dq_from_ik_info(
        env,
        ik_info,
        stepsize       = 1,
        eps            = 1e-2,
        th             = np.radians(1.0),
        joint_idxs_jac = None,
    ):
    """
        Get delta q from augmented Jacobian method
    """
    J_list,ik_err_list = [],[]
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None
        J,ik_err = env.get_ik_ingredients(
            body_name = ik_body_name,
            geom_name = ik_geom_name,
            p_trgt    = ik_p_trgt,
            R_trgt    = ik_R_trgt,
            IK_P      = IK_P,
            IK_R      = IK_R,
        )
        J_list.append(J)
        ik_err_list.append(ik_err)

    J_stack      = np.vstack(J_list)
    ik_err_stack = np.hstack(ik_err_list)

    # Select Jacobian columns that are within the joints to use
    if joint_idxs_jac is not None:
        J_stack_backup = J_stack.copy()
        J_stack = np.zeros_like(J_stack)
        J_stack[:,joint_idxs_jac] = J_stack_backup[:,joint_idxs_jac]

    # Compute dq from damped least square
    dq = env.damped_ls(J_stack,ik_err_stack,stepsize=stepsize,eps=eps,th=th)
    return dq,ik_err_stack

def plot_ik_info(
        env,
        ik_info,
        axis_len   = 0.05,
        axis_width = 0.005,
        sphere_r   = 0.01,
        ):
    """
        Plot IK information
    """
    colors = get_colors(cmap_name='gist_rainbow',n_color=ik_info['n_trgt'])
    for ik_idx,(ik_body_name,ik_geom_name) in enumerate(zip(ik_info['body_names'],ik_info['geom_names'])):
        color = colors[ik_idx]
        ik_p_trgt = ik_info['p_trgts'][ik_idx]
        ik_R_trgt = ik_info['R_trgts'][ik_idx]
        IK_P = ik_p_trgt is not None
        IK_R = ik_R_trgt is not None

        if ik_body_name is not None:
            # Plot current 
            env.plot_body_T(
                body_name   = ik_body_name,
                plot_axis   = IK_R,
                axis_len    = axis_len,
                axis_width  = axis_width,
                plot_sphere = IK_P,
                sphere_r    = sphere_r,
                sphere_rgba = color,
                label       = '' # ''/ik_body_name
            )
            # Plot target
            if IK_P:
                env.plot_sphere(p=ik_p_trgt,r=sphere_r,rgba=color,label='') 
                env.plot_line_fr2to(p_fr=env.get_p_body(body_name=ik_body_name),p_to=ik_p_trgt,rgba=color)
            if IK_P and IK_R:
                env.plot_T(p=ik_p_trgt,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            if not IK_P and IK_R: # rotation only
                p_curr = env.get_p_body(body_name=ik_body_name)
                env.plot_T(p=p_curr,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            
        if ik_geom_name is not None:
            # Plot current 
            env.plot_geom_T(
                geom_name   = ik_geom_name,
                plot_axis   = IK_R,
                axis_len    = axis_len,
                axis_width  = axis_width,
                plot_sphere = IK_P,
                sphere_r    = sphere_r,
                sphere_rgba = color,
                label       = '' # ''/ik_geom_name
            )
            # Plot target
            if IK_P:
                env.plot_sphere(p=ik_p_trgt,r=sphere_r,rgba=color,label='') 
                env.plot_line_fr2to(p_fr=env.get_p_geom(geom_name=ik_geom_name),p_to=ik_p_trgt,rgba=color)
            if IK_P and IK_R:
                env.plot_T(p=ik_p_trgt,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)
            if not IK_P and IK_R: # rotation only
                p_curr = env.get_p_geom(geom_name=ik_geom_name)
                env.plot_T(p=p_curr,R=ik_R_trgt,plot_axis=True,axis_len=axis_len,axis_width=axis_width)

def solve_ik(
        env,
        joint_names_for_ik,
        body_name_trgt,
        q_init          = None, # IK start from the initial pose
        p_trgt          = None,
        R_trgt          = None,
        max_ik_tick     = 1000,
        ik_err_th       = 1e-2,
        restore_state   = True,
        ik_stepsize     = 1.0,
        ik_eps          = 1e-2,
        ik_th           = np.radians(1.0),
        verbose         = False,
        verbose_warning = True,
        reset_env       = False,
        render          = False,
        render_every    = 1,
    ):
    """ 
        Solve Inverse Kinematics
    """
    # Reset
    if reset_env:
        env.reset()
    if render:
        env.init_viewer()
    # Joint indices
    joint_idxs_jac = env.get_idxs_jac(joint_names=joint_names_for_ik)
    joint_idxs_fwd = env.get_idxs_fwd(joint_names=joint_names_for_ik)
    # Joint range
    q_mins = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),0]
    q_maxs = env.joint_ranges[get_idxs(env.joint_names,joint_names_for_ik),1]
    # Store MuJoCo state
    if restore_state:
        env.store_state()
    # Initial IK pose
    if q_init is not None:
        env.forward(q=q_init,joint_idxs=joint_idxs_fwd,increase_tick=False)
    # Initialize IK information
    ik_info = init_ik_info()
    add_ik_info(
        ik_info  = ik_info,
        body_name= body_name_trgt,
        p_trgt   = p_trgt,
        R_trgt   = R_trgt, 
    )
    # Loop
    q_curr = env.get_qpos_joints(joint_names=joint_names_for_ik)
    for ik_tick in range(max_ik_tick):
        dq,ik_err_stack = get_dq_from_ik_info(
            env            = env,
            ik_info        = ik_info,
            stepsize       = ik_stepsize,
            eps            = ik_eps,
            th             = ik_th,
            joint_idxs_jac = joint_idxs_jac,
        )
        q_curr = q_curr + dq[joint_idxs_jac] # update
        q_curr = np.clip(q_curr,q_mins,q_maxs) # clip
        env.forward(q=q_curr,joint_idxs=joint_idxs_fwd,increase_tick=False) # fk
        ik_err = np.linalg.norm(ik_err_stack) # IK error
        if ik_err < ik_err_th: break # terminate condition
        if verbose:
            print ("[%d/%d] ik_err:[%.3f]"%(ik_tick,max_ik_tick,ik_err))
        if render:
            if ik_tick%render_every==0:
                plot_ik_info(env,ik_info)
                env.render()
    # Print if IK error is too high
    if verbose_warning and ik_err > ik_err_th:
        print ("ik_err:[%.4f] is higher than ik_err_th:[%.4f]."%
               (ik_err,ik_err_th))
        print ("You may want to increase max_ik_tick:[%d]"%
               (max_ik_tick))
    # Restore backuped state
    if restore_state:
        env.restore_state()
    # Close viewer
    if render:
        env.close_viewer()
    # Return
    return q_curr,ik_err_stack,ik_info
