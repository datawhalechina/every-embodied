import sys
import io
import cv2
sys.path.append('./external-libraries')
import airsim
import math
import numpy as np
import random
import torch
from torchvision.ops import box_convert
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict, annotate

BOX_TRESHOLD = 0.25
TEXT_TRESHOLD = 0.25


objects_dict = {}

class AirSimWrapper:
    def __init__(self,ip=""):
        if ip == "":
            self.client = airsim.MultirotorClient()
        else:
            self.client = airsim.MultirotorClient(ip=ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)


        #图像信息
        self.image_data = "" #一般图像数据
        self.img_depth_planar = "" #深度图像，到相平面
        self.img_depth_perspective = "" #到相机

        self.img = "" #open cv图像

        #目标检测
        self.dino_model = load_model("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth")

    def takeoff(self):
        self.client.takeoffAsync().join()

    def land(self):
        self.client.landAsync().join()

    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        yaw_degree = self.get_yaw()  # angle in degree
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val,yaw_degree]

    def fly_to(self, point):
        if point[2] > 0:
            self.client.moveToPositionAsync(point[0], point[1], -point[2], 1).join()
        else:
            self.client.moveToPositionAsync(point[0], point[1], point[2], 1).join()


    def fly_path(self, points):
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()

    def set_yaw(self, yaw):
        self.client.rotateToYawAsync(yaw, 5).join()

    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2] # get the yaw angle
        yaw_degree = math.degrees(yaw)
        return yaw_degree # return the yaw angle in degree

    def get_position(self, object_name):
        query_string = objects_dict[object_name] + ".*"
        object_names_ue = []
        while len(object_names_ue) == 0:
            object_names_ue = self.client.simListSceneObjects(query_string)
        pose = self.client.simGetObjectPose(object_names_ue[0])
        yaw_degree = math.degrees(pose.orientation.z_val) #angle in degree
        
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val, yaw_degree]

    def reset(self):
        self.client.reset()

    def get_distance(self):
        distance = 100000000

        pose = self.client.simGetVehiclePose()  # get the current pose of the quadcopter
        v_p = [pose.position.x_val, pose.position.y_val, pose.position.z_val]

        # get lidar data
        lidarData = self.client.getLidarData()
        if len(lidarData.point_cloud) < 3:
            return distance # if no points are received from the lidar, return a big distance as 100000000

        points = np.array(lidarData.point_cloud, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        distance_list = []
        for p in points:
            distance = np.linalg.norm(np.array(v_p) - p)
            distance_list.append(distance)

        distance = min(distance_list)
        return distance

    def look_at(self, yaw_degree):
        self.set_yaw(yaw_degree)


    def turn(self, angle):
        """
        旋转angle
        :return:
        """
        yaw_degree = self.get_yaw()
        yaw_degree = yaw_degree + angle
        self.set_yaw(yaw_degree)

    def move(self, distance):
        """
        向前移动distance距离
        :return:
        """
        step_length = distance
        cur_position = self.get_drone_position()
        yaw_degree = cur_position[3]
        #将角度转换为弧度
        yaw = math.radians(yaw_degree)
        #向前移动0.1米
        x = cur_position[0] + step_length*math.cos(yaw)
        y = cur_position[1] + step_length*math.sin(yaw)
        z = cur_position[2]
        self.fly_to([x, y, z])

        

    def turn_left(self):
        """
        左转10度
        :return:
        """
        yaw_degree = self.get_yaw()
        yaw_degree = yaw_degree - 10
        self.set_yaw(yaw_degree)


    def turn_right(self):
        """
        右转10度
        :return:
        """
        yaw_degree = self.get_yaw()
        yaw_degree = yaw_degree + 10
        self.set_yaw(yaw_degree)

    def forward(self):
        """
        向前移动1米, 太少了不动
        :return:
        """
        step_length = 1
        cur_position = self.get_drone_position()
        yaw_degree = cur_position[3]
        #将角度转换为弧度
        yaw = math.radians(yaw_degree)
        #向前移动0.1米
        x = cur_position[0] + step_length*math.cos(yaw)
        y = cur_position[1] + step_length*math.sin(yaw)
        z = cur_position[2]
        self.fly_to([x, y, z])

    def get_image(self):
        """
        获得前置摄像头渲染图像
        :return:
        """
        responses = self.client.simGetImages([
            # png format
            airsim.ImageRequest(0, airsim.ImageType.Scene, pixels_as_float=False, compress=True),

            # floating point uncompressed image，深度图, 像素点代表到相平面距离
            airsim.ImageRequest(0, airsim.ImageType.DepthPlanar, pixels_as_float=True),

            # 像素点代表的到相机的距离
            airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, pixels_as_float=True)
          ]
        )

        self.image_data = responses[0].image_data_uint8
        self.img_depth_planar = np.array(responses[1].image_data_float).reshape(responses[1].height, responses[0].width)
        self.img_depth_perspective = np.array(responses[2].image_data_float).reshape(responses[2].height, responses[1].width)
        img = cv2.imdecode(np.array(bytearray(self.image_data), dtype='uint8'), cv2.IMREAD_UNCHANGED)  # 从二进制图片数据中读
        self.img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # 4通道转3

        return self.img

    def ob_objects(self,obj_name_list):
        """
        注意需要先执行get_image，
        在图像 img 上运行对象检测模型，获得目标列表 [ <对象名称、距离、角度（以度为单位）>,...]
        :return:对象名称列表、对象信息列表、bbox图
        """

        TEXT_PROMPT = " | ".join(obj_name_list)
        #目标检测
        imgbytes = cv2.imencode(".jpg", self.img)[1].tobytes()
        byte_stream = io.BytesIO(imgbytes)
        
    
        image_source, image = load_image(byte_stream)
        
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image, 
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        h, w, _ = image_source.shape
        boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
        boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        
        #[xmin, ymin, xmax, ymax]
        obj_locs = boxes_xyxy


        final_obj_list = [] #最终结果列表
        #构建目标结果
        index = 0
        for bbox in obj_locs:
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)

            depth_distance = self.img_depth_planar[center_y, center_x, ] #相平面距离
            camera_distance = self.img_depth_perspective[center_y, center_x] #相机距离

            #求角度
            angel = math.acos(depth_distance / camera_distance)
            angel_degree = math.degrees(angel)

            # 判断正负，左边为正，右边为负，只看偏航角
            if center_x < self.img.shape[1] / 2:
                # 如果目标在图像的左侧，向左转，degree 为负数
                angel_degree = -1 * angel_degree

            obj_name =  phrases[index]#获得目标名称，可能有多个

            obj_info = (obj_name, camera_distance, depth_distance, angel_degree, center_x, center_y)
            final_obj_list.append(obj_info)
            index = index + 1

        #画框
        #annotated_frame：cv2的图片，image_source：pil图片
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        final_result = []

        for obj_info in final_obj_list:
            item = (obj_info[0], obj_info[1], obj_info[3]) #obj_name, camera_distance, angel_degree
            final_result.append(item)

        return final_result


    def ob_objects_llm(self,obj_name_list):
        """
        注意需要先执行get_image，为llm提供观测结果
        在图像 img 上运行对象检测模型，获得目标列表 [ <对象名称、距离、角度（以度为单位）>,...] , 给到llm用于推理
        :return:[ <对象名称、距离、角度（以度为单位）>,...] 如[(门，0.53，22)，(椅子，4.84，-21)]
        """
        #获得识别结果
        ob_list, final_obj_list, annotated_frame = self.ob_objects(obj_name_list)

        final_result = []

        for obj_info in final_obj_list:
            item = (obj_info[0], obj_info[1], obj_info[3]) #obj_name, camera_distance, angel_degree
            final_result.append(item)

        return final_result

        
        