#!/usr/bin/env python3

import cv2
import cv2.aruco as aruco
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from cv_bridge import CvBridge
import tf
import tf.transformations as tf_trans
import tf2_ros
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
from utilities import quaternion_multiply


class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_detector', anonymous=True)
        self.left_image_sub = rospy.Subscriber('/left_camera/color/image_raw', Image, self.left_image_callback)
        self.right_image_sub = rospy.Subscriber('/right_camera/color/image_raw', Image, self.right_image_callback)
        
        self.left_camera_info_sub = rospy.Subscriber('/left_camera/color/camera_info', CameraInfo, self.left_camera_info_callback)
        self.right_camera_info_sub = rospy.Subscriber('/right_camera/color/camera_info', CameraInfo, self.right_camera_info_callback)
        
        self.T_left_to_base = np.loadtxt('/home/riot/kinova_gen3_lite/src/box_detection/cam_poses/camera_pose_left.txt', delimiter=' ')
        self.T_right_to_base = np.loadtxt('/home/riot/kinova_gen3_lite/src/box_detection/cam_poses/camera_pose_right.txt', delimiter=' ')
        
        
        self.pose_pub = rospy.Publisher('/aruco_poses', PoseArray, queue_size=10)
        self.box_pub = rospy.Publisher('/box_pose', PoseStamped, queue_size=10)
        self.bridge = CvBridge()
        
        self.left_camera_matrix = None
        self.left_dist_coeffs = None
        
        self.right_camera_matrix = None
        self.right_dist_coeffs = None
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.parameters = aruco.DetectorParameters()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.listener = tf.TransformListener()
        self.left_fused_poses = None
        self.right_fused_poses = None
        # Predefined transformations from each marker to the box center
        self.left_marker_to_box_transforms = {
            10: {'translation': [0.043, 0.001, -0.026], 'rotation': [0.0, 0.0, 0.0, 1.0]},
            9: {'translation': [-0.043, 0.002, -0.026], 'rotation': [0.0, 0.0, 0.0, 1.0]},
            3: {'translation': [0.0035, 0.046, 0], 'rotation': [0.0, 0.0, 0.0, 1.0]},
            4: {'translation': [-0.072, -0.0035, 0], 'rotation': [0.0, 0.0, 0.0, 1.0]},
            6: {'translation': [0.0035, -0.046, 0], 'rotation': [0.0, 0.0, 0.0, 1.0]},
            7: {'translation': [0.072, -0.0035, 0], 'rotation': [0.0, 0.0, 0.0, 1.0]},
            # # Add more markers as needed
        }

        # Predefined transformations from each marker to the box center
        self.right_marker_to_box_transforms = {
            10: {'translation': [-0.043, 0.002, -0.026], 'rotation': [0.0, 0.0, 0.0, 1.0]},
            9: {'translation': [0.043, 0.001, -0.026], 'rotation': [0.0, 0.0, 0.0, 1.0]},
            # Add more markers as needed
        }

        
        # Initialize a timer to call the publish_box_pose method
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)  # Adjust the duration as needed



    def left_camera_info_callback(self, msg):
        self.left_camera_matrix = np.array(msg.K).reshape(3, 3)
        self.left_dist_coeffs = np.array(msg.D)
    
    def right_camera_info_callback(self, msg):
        self.right_camera_matrix = np.array(msg.K).reshape(3, 3)
        self.right_dist_coeffs = np.array(msg.D)

    def left_image_callback(self, data):
        
        if self.left_camera_matrix is None or self.left_dist_coeffs is None:
            return
        
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.left_camera_matrix, self.left_dist_coeffs)
            pose_array = PoseArray()
            pose_array.header.stamp = rospy.Time.now()
            pose_array.header.frame_id = 'left_camera_color_optical_frame'
            detected_marker_poses = []

            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # Convert rvec and tvec to transformation matrix
                rmat, _ = cv2.Rodrigues(rvec)
                tmat = np.column_stack((rmat, tvec))
                tmat = np.row_stack((tmat, [0, 0, 0, 1]))

                
                # Create Pose message 
                pose = Pose()
                pose.position.x = tvec[0]
                pose.position.y = tvec[1]
                pose.position.z = tvec[2]
                quat = tf_trans.quaternion_from_matrix(tmat)
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]


                 # Create TransformStamped message
                transform = TransformStamped()
                transform.header.stamp = rospy.Time.now()
                transform.header.frame_id = 'left_camera_color_optical_frame'
                transform.child_frame_id = f'arucomarker_{marker_id}_left'
                transform.transform.translation.x = tvec[0]
                transform.transform.translation.y = tvec[1]
                transform.transform.translation.z = tvec[2]
                transform.transform.rotation.x = quat[0]
                transform.transform.rotation.y = quat[1]
                transform.transform.rotation.z = quat[2]
                transform.transform.rotation.w = quat[3]

                # Broadcast the transform
                self.tf_broadcaster.sendTransform(transform)
                
                
                
                # Apply the static transformation to get the box's pose from this marker
                if marker_id in self.left_marker_to_box_transforms:
                    marker_to_box = self.left_marker_to_box_transforms[marker_id]
                    # This pose you got is from robot_base to box_frame
                    box_pose = self.apply_static_transform(pose, marker_to_box, 'left_camera_color_optical_frame', 'base_link')
                    detected_marker_poses.append(box_pose)
                    pose_array.poses.append(pose)
                    self.pose_pub.publish(pose_array)
            if detected_marker_poses:
                self.left_fused_poses = self.fuse_poses(detected_marker_poses)
                
    
            
            # Visualize the pose
            # self.box_pub.publish(pose)
            # # Publish the tf of the box
            # self.publish_fused_box_pose(pose)
                
                
                
    def right_image_callback(self, data):
        if self.right_camera_matrix is None or self.right_dist_coeffs is None:
            return
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.right_camera_matrix, self.right_dist_coeffs)
            pose_array = PoseArray()
            pose_array.header.stamp = rospy.Time.now()
            pose_array.header.frame_id = 'right_camera_color_optical_frame'
            detected_marker_poses = []

            for i, marker_id in enumerate(ids.flatten()):
                rvec = rvecs[i][0]
                tvec = tvecs[i][0]

                # Convert rvec and tvec to transformation matrix
                rmat, _ = cv2.Rodrigues(rvec)
                tmat = np.column_stack((rmat, tvec))
                tmat = np.row_stack((tmat, [0, 0, 0, 1]))



                # Create Pose message
                pose = Pose()
                pose.position.x = tvec[0]
                pose.position.y = tvec[1]
                pose.position.z = tvec[2]
                quat = tf_trans.quaternion_from_matrix(tmat)
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]



                # Create TransformStamped message
                transform = TransformStamped()
                transform.header.stamp = rospy.Time.now()
                transform.header.frame_id = 'right_camera_color_optical_frame'
                transform.child_frame_id = f'arucomarker_{marker_id}_right'
                transform.transform.translation.x = tvec[0]
                transform.transform.translation.y = tvec[1]
                transform.transform.translation.z = tvec[2]
                transform.transform.rotation.x = quat[0]
                transform.transform.rotation.y = quat[1]
                transform.transform.rotation.z = quat[2]
                transform.transform.rotation.w = quat[3]

                # Broadcast the transform
                self.tf_broadcaster.sendTransform(transform)















                # Apply the static transformation to get the box's pose from this marker
                if marker_id in self.right_marker_to_box_transforms:
                    marker_to_box = self.right_marker_to_box_transforms[marker_id]
                    # This pose you got is from robot_base to box_frame
                    box_pose = self.apply_static_transform(pose, marker_to_box, 'right_camera_color_optical_frame', 'base_link')
                    detected_marker_poses.append(box_pose)
                    # self.pose_pub.publish(pose_array)
            # Publish fused box pose
            if detected_marker_poses:
                self.right_fused_poses = self.fuse_poses(detected_marker_poses)
                
               
                
                
                
                
    def timer_callback(self, event):
        if self.left_fused_poses is not None or self.right_fused_poses is not None:
            combined_pose = self.fuse_poses([self.left_fused_poses, self.right_fused_poses])
            
            shift = np.loadtxt('/home/riot/kinova_gen3_lite/src/box_detection/cam_poses/box_frame_shift.txt', delimiter=' ')
            shift_quat = quaternion_multiply(combined_pose.orientation, shift)
            
            combined_pose.orientation.x = shift_quat[0]
            combined_pose.orientation.y = shift_quat[1]
            combined_pose.orientation.z = shift_quat[2]
            combined_pose.orientation.w = shift_quat[3]
            
            
            
            # Visualize the box's frame
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = 'base_link'
            pose.pose = combined_pose
            
            # Visualize the pose
            self.box_pub.publish(pose)
            # Publish the tf of the box
            self.publish_fused_box_pose(pose)
                    

        
    def apply_static_transform(self, pose: Pose, marker_to_box_transform, camera_frame: str, robot_frame: str):
        # Convert pose to transformation matrix
        pose_matrix = tf_trans.quaternion_matrix([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        pose_matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

        # Convert marker_to_box transform to transformation matrix
        marker_to_box_matrix = tf_trans.quaternion_matrix(marker_to_box_transform['rotation'])
        marker_to_box_matrix[:3, 3] = marker_to_box_transform['translation']
        
        # Apply static transformation
        box_pose_matrix = np.dot(pose_matrix, marker_to_box_matrix)
        
        # Get transformation from base_link to camera_optical_frame
        self.listener.waitForTransform(robot_frame, camera_frame, rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = self.listener.lookupTransform(robot_frame, camera_frame, rospy.Time(0))

        # Convert base_link to camera_optical_frame transformation to matrix
        base_to_camera_matrix = tf_trans.quaternion_matrix(rot)
        base_to_camera_matrix[:3, 3] = trans

        # Apply transformation from camera_optical_frame to base_link
        result_matrix = np.dot(base_to_camera_matrix, box_pose_matrix)


        # Convert back to Pose
        result_pose = Pose()
        result_pose.position.x = result_matrix[0, 3]
        result_pose.position.y = result_matrix[1, 3]
        result_pose.position.z = result_matrix[2, 3]
        result_quat = tf_trans.quaternion_from_matrix(result_matrix)
        result_pose.orientation.x = result_quat[0]
        result_pose.orientation.y = result_quat[1]
        result_pose.orientation.z = result_quat[2]
        result_pose.orientation.w = result_quat[3]

        return result_pose

    def fuse_poses(self, poses):
        # Simple averaging for translation
        translations = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in poses])
        avg_translation = np.mean(translations, axis=0)

        # Simple averaging for rotation (quaternion)
        quaternions = np.array([[pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] for pose in poses])
        avg_quaternion = np.mean(quaternions, axis=0)
        avg_quaternion /= np.linalg.norm(avg_quaternion)  # Normalize quaternion

        avg_pose = Pose()
        avg_pose.position.x = avg_translation[0]
        avg_pose.position.y = avg_translation[1]
        avg_pose.position.z = avg_translation[2]
        avg_pose.orientation.x = avg_quaternion[0]
        avg_pose.orientation.y = avg_quaternion[1]
        avg_pose.orientation.z = avg_quaternion[2]
        avg_pose.orientation.w = avg_quaternion[3]

        return avg_pose

    def publish_fused_box_pose(self, box_pose):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = 'base_link'
        transform.child_frame_id = 'box_frame'
        transform.transform.translation.x = box_pose.pose.position.x
        transform.transform.translation.y = box_pose.pose.position.y
        transform.transform.translation.z = box_pose.pose.position.z
        transform.transform.rotation.x = box_pose.pose.orientation.x
        transform.transform.rotation.y = box_pose.pose.orientation.y
        transform.transform.rotation.z = box_pose.pose.orientation.z
        transform.transform.rotation.w = box_pose.pose.orientation.w
        self.tf_broadcaster.sendTransform(transform)

if __name__ == '__main__':
    try:
        aruco_detector = ArucoDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
