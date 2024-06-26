#!/usr/bin/env python3

import rospy
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from utilities import *


class PointCloudCombiner:
    def __init__(self):
        rospy.init_node('pointcloud_combiner')

        self.pcd1 = None
        self.pcd2 = None

        self.sub_left = rospy.Subscriber('/left_camera/depth/color/points', PointCloud2, self.callback_left)
        self.sub_right = rospy.Subscriber('/right_camera/depth/color/points', PointCloud2, self.callback_right)

        self.pub_combined = rospy.Publisher('/combined_pointcloud', PointCloud2, queue_size=1)

        self.transformation_matrix_1 = np.loadtxt('/home/riot/kinova_gen3_lite/src/heightmap/real/camera_pose_left.txt', delimiter=' ')
        self.transformation_matrix_2 = np.loadtxt('/home/riot/kinova_gen3_lite/src/heightmap/real/camera_pose_right.txt', delimiter=' ')

    def callback_left(self, msg):
        self.pcd1 = convertCloudFromRosToOpen3d(msg)

    def callback_right(self, msg):
        self.pcd2 = convertCloudFromRosToOpen3d(msg)

    def convert_ros_to_o3d(self, ros_point_cloud):
        # Extract point cloud data from the ROS message
        points = np.array(list(pc2.read_points(ros_point_cloud, skip_nans=True, field_names=("x", "y", "z"))))

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        return pcd

    def convert_o3d_to_ros(self, o3d_point_cloud, frame_id):
        points = np.asarray(o3d_point_cloud.points)

        # Create the PointCloud2 message
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        cloud_data = np.zeros(points.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ])

        cloud_data['x'] = points[:, 0]
        cloud_data['y'] = points[:, 1]
        cloud_data['z'] = points[:, 2]

        return pc2.create_cloud(header, [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
        ], cloud_data)

    def process_pointclouds(self):
        if self.pcd1 is not None and self.pcd2 is not None:
            # Apply known transformations
            self.pcd1.transform(self.transformation_matrix_1)
            self.pcd2.transform(self.transformation_matrix_2)

            # Aligning pointclouds obtained from two cameras
            threshold = 0.02  # Distance threshold
            trans_init = np.identity(4)  # Initial transformation matrix

            # Apply ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                self.pcd1, self.pcd2, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())

            # Transform pcd1 to align with pcd2
            self.pcd1.transform(reg_p2p.transformation)

            # Combine point clouds
            combined_pcd = self.pcd1 + self.pcd2

            # Publish the combined point cloud
            combined_ros_pcd = convertCloudFromOpen3dToRos(combined_pcd, 'base_link')  # Adjust frame_id as necessary
            self.pub_combined.publish(combined_ros_pcd)



if __name__ == '__main__':
    combiner = PointCloudCombiner()
    rate = rospy.Rate(100)  # 10 Hz
    while not rospy.is_shutdown():
        combiner.process_pointclouds()
        rate.sleep()
