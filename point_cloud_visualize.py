import numpy as np
import open3d as o3d
import time
import os
import cv2
import argparse

from matplotlib import pyplot as plt


def main(args):
    label_path = args.label_path
    input_path = args.input_path
    hide_window = args.hide_window == None
    video_output_path = args.video_output_path

    LABEL_KEYS = np.load(label_path)

    LABEL_COLORS = {
      0 : [0, 0, 0],
      1 : [0, 0, 255],
      10: [245, 150, 100],
      11: [245, 230, 100],
      13: [250, 80, 100],
      15: [150, 60, 30],
      16: [255, 0, 0],
      18: [180, 30, 80],
      20: [255, 0, 0],
      30: [30, 30, 255],
      31: [200, 40, 255],
      32: [90, 30, 150],
      40: [255, 0, 255],
      44: [255, 150, 255],
      48: [75, 0, 75],
      49: [75, 0, 175],
      50: [0, 200, 255],
      51: [50, 120, 255],
      52: [0, 150, 255],
      60: [170, 255, 150],
      70: [0, 175, 0],
      71: [0, 60, 135],
      72: [80, 240, 150],
      80: [150, 240, 255],
      81: [0, 0, 255],
      99: [255, 255, 50],
      252: [245, 150, 100],
      256: [255, 0, 0],
      253: [200, 40, 255],
      254: [30, 30, 255],
      255: [90, 30, 150],
      257: [250, 80, 100],
      258: [180, 30, 80],
      259: [255, 0, 0],
    }

    LEARNING_MAP = {
      0: 0,      # "unlabeled", and others ignored
      1: 10,     # "car"
      2: 11,     # "bicycle"
      3: 15,     # "motorcycle"
      4: 18,     # "truck"
      5: 20,     # "other-vehicle"
      6: 30,     # "person"
      7: 31,     # "bicyclist"
      8: 32,     # "motorcyclist"
      9: 40,     # "road"
      10: 44,    # "parking"
      11: 48,    # "sidewalk"
      12: 49,    # "other-ground"
      13: 50,    # "building"
      14: 51,    # "fence"
      15: 70,    # "vegetation"
      16: 71,    # "trunk"
      17: 72,    # "terrain"
      18: 80,    # "pole"
      19: 81,    # "traffic-sign"
    }

    color_per_label = {}

    for i in range(len(LABEL_KEYS) + 1):
        color_per_label[i] = LABEL_COLORS[LEARNING_MAP[i]]
        color_per_label[i].reverse()





    def find_colors(labels):
        result = np.zeros((len(labels), 3))
        for i in range(len(labels)):
            curr_color = LABEL_COLORS[LEARNING_MAP[labels[i]]]
            result[i][0] = curr_color[0]
            result[i][1] = curr_color[1]
            result[i][2] = curr_color[2]
        return result

    def update_point_cloud(file, pcd):
        example = np.load(file, allow_pickle=True)

        point_cloud_vals = example[:, :3] # point cloud locations
        point_cloud_labels_pred = example[:, 3] # labels

        colors = find_colors(list(point_cloud_labels_pred)) / 255


        pcd.points = o3d.utility.Vector3dVector(point_cloud_vals)
        pcd.colors =  o3d.utility.Vector3dVector(colors)

    def add_point_cloud_to_scene(vis, pcd):
        # mat = o3d.visualization.rendering.MaterialRecord()
        # mat.shader = 'defaultUnlit'
        # mat.point_size = 9.0

        vis.add_geometry(pcd)

    def update_visualizer_view():
        camera_control = vis.get_view_control()
        camera_control.set_lookat(( 3.0695789143973324, 1.0532306959878019, 2.9624452627806455 ))
        camera_control.set_up(( 0.36022430259164362, 0.011528896318685228, 0.93279447702697993 ))
        camera_control.set_front(( -0.90028904687379541, -0.25764288267730373, 0.35085577818357544 ))
        camera_control.set_zoom(0.080000000000000002)
        camera_control.change_field_of_view(60)

        render_control = vis.get_render_option()
        render_control.background_color = np.asarray([0, 0, 0])


    total_examples = [file for file in os.listdir(input_path) if file.startswith("vals_")]
    num_examples = len(total_examples)
        
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=700, width=700, visible=hide_window)

    pcd = o3d.geometry.PointCloud()
    update_point_cloud("vals_0", pcd)
    add_point_cloud_to_scene(vis, pcd)

    keep_running = True

    fps_rate = 1e9/30
    previous_time_clock = time.time_ns()
    current = 0

    update_visualizer_view()

    imgs = []

    if (video_output_path):
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(hide_window)
        imgs.append(img)

    while(keep_running):
        if (not hide_window or time.time_ns() - previous_time_clock >= fps_rate):
            # update
            current = ((current + 1) % num_examples)

            if video_output_path and current == 0:
                break

            update_point_cloud("%s/vals_%d" % (input_path, current), pcd)
            vis.update_geometry(pcd)

            previous_time_clock = time.time_ns()

            update_visualizer_view()
            if (video_output_path):
                img = vis.capture_screen_float_buffer(hide_window)
                imgs.append(img)
                vis.update_renderer()

        keep_running = vis.poll_events()


    vis.destroy_window()

    fps = 10

    bounds = imgs[0].get_max_bound() - imgs[0].get_min_bound()
    width = int(bounds[0])
    height = int(bounds[1])

    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
    for img in imgs:
        # cv2.imshow("Vals", np.uint8(np.asarray(img)* 255))
        # cv2.waitKey(0)
        img_processed = np.uint8(np.asarray(img) * 255)
        final_img = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
        out.write(final_img)
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates a visualization of the inputs and outputs of semantic segmentation for point clouds.')
    parser.add_argument('-v', '--video_output_path', default='')
    parser.add_argument('-w', '--hide_window', action=argparse.BooleanOptionalAction)
    parser.add_argument('-i', '--input_path', default='demo_results/')
    parser.add_argument('-l', '--label_path', default='demo_results/label_vals.npy')
    args = parser.parse_args()

    main(args)

"""
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 79.004719171669677, 78.367078918044129, 2.9560461044311523 ],
			"boundingbox_min" : [ -79.602406925084267, -79.916901006196852, -27.639139175415039 ],
			"field_of_view" : 60.0,
			"front" : [ -0.90028904687379541, -0.25764288267730373, 0.35085577818357544 ],
			"lookat" : [ 3.0695789143973324, 1.0532306959878019, 2.9624452627806455 ],
			"up" : [ 0.36022430259164362, 0.011528896318685228, 0.93279447702697993 ],
			"zoom" : 0.080000000000000002
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

"""
