import click
import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
import mediapipe as mp
import skimage
from skimage.transform import PiecewiseAffineTransform, warp

def normalize_keypoints(keypoints3d):
	center = keypoints3d[0]
	keypoints3d = keypoints3d - center
	axis1 = keypoints3d[165] - keypoints3d[391]
	axis2 = keypoints3d[2] - keypoints3d[0]
	axis3 = np.cross(axis2,axis1)
	axis3 = axis3/np.linalg.norm(axis3)
	axis2 = axis2/np.linalg.norm(axis2)
	axis1 = np.cross(axis3, axis2)
	axis1 = axis1/np.linalg.norm(axis1)
	U = np.array([axis3,axis2,axis1])
	keypoints3d = keypoints3d.dot(U)
	keypoints3d = keypoints3d - keypoints3d.mean(axis=0)
	return keypoints3d

	# borrowed from https://github.com/YadiraF/DECA/blob/f84855abf9f6956fb79f3588258621b363fa282c/decalib/utils/util.py
def load_obj(filepath):
	""" Ref: https://github.com/facebookresearch/pytorch3d/blob/25c065e9dafa90163e7cec873dbb324a637c68b7/pytorch3d/io/obj_io.py
	Load a mesh from a file-like object.
	"""
	with open(filepath, 'r') as file:
		lines = [line.strip() for line in file]

	vertices = []
	uv_coords = []
	faces = []
	uv_faces = []
	
	# startswith expects each line to be a string. If the file is read in as
	# bytes then first decode to strings.
	if lines and isinstance(lines[0], bytes):
		lines = [el.decode("utf-8") for el in lines]

	for line in lines:
		tokens = line.strip().split()

		# parse vertex
		if tokens[0] == "v":
			vertices.append([float(x) for x in tokens[1:4]])
			continue
		# parse uv coord
		if tokens[0] == "vt":
			uv_coords.append([float(x) for x in tokens[1:3]])
			continue
		# parse face
		if tokens[0] == "f":
			for vertex in [_face.split("/") for _face in tokens[1:]]:
				faces.append(int(vertex[0]))
				if len(vertex) > 1 and vertex[1] != "":
					uv_faces.append(int(vertex[1]))
			continue

		return (
				np.array(vertices),
				np.array(uv_coords),
				np.array(faces).reshape(-1, 3) - 1,
				np.array(uv_faces).reshape(-1, 3) - 1
		)

# borrowed from https://github.com/YadiraF/DECA/blob/f84855abf9f6956fb79f3588258621b363fa282c/decalib/utils/util.py
def write_obj(obj_name,
							vertices,
							faces,
							texture_name = "texture.jpg",
							colors=None,
							texture=None,
							uvcoords=None,
							uvfaces=None,
							inverse_face_order=False,
							normal_map=None,
							):
		''' Save 3D face model with texture. 
		Ref: https://github.com/patrikhuber/eos/blob/bd00155ebae4b1a13b08bf5a991694d682abbada/include/eos/core/Mesh.hpp
		Args:
				obj_name: str
				vertices: shape = (nver, 3)
				colors: shape = (nver, 3)
				faces: shape = (ntri, 3)
				texture: shape = (uv_size, uv_size, 3)
				uvcoords: shape = (nver, 2) max value<=1
		'''
		if os.path.splitext(obj_name)[-1] != '.obj':
				obj_name = obj_name + '.obj'
		mtl_name = obj_name.replace('.obj', '.mtl')
		texture_name
		material_name = 'FaceTexture'

		faces = faces.copy()
		# mesh lab start with 1, python/c++ start from 0
		faces += 1
		if inverse_face_order:
				faces = faces[:, [2, 1, 0]]
				if uvfaces is not None:
						uvfaces = uvfaces[:, [2, 1, 0]]

		# write obj
		with open(obj_name, 'w') as f:
				# first line: write mtlib(material library)
				# f.write('# %s\n' % os.path.basename(obj_name))
				# f.write('#\n')
				# f.write('\n')
				if texture is not None:
						f.write('mtllib %s\n\n' % os.path.basename(mtl_name))

				# write vertices
				if colors is None:
						for i in range(vertices.shape[0]):
								f.write('v {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
				else:
						for i in range(vertices.shape[0]):
								f.write('v {} {} {} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

				# write uv coords
				if texture is None:
						for i in range(faces.shape[0]):
								f.write('f {} {} {}\n'.format(faces[i, 2], faces[i, 1], faces[i, 0]))
				else:
						for i in range(uvcoords.shape[0]):
								f.write('vt {} {}\n'.format(uvcoords[i,0], uvcoords[i,1]))
						f.write('usemtl %s\n' % material_name)
						# write f: ver ind/ uv ind
						uvfaces = uvfaces + 1
						for i in range(faces.shape[0]):
								f.write('f {}/{} {}/{} {}/{}\n'.format(
										#  faces[i, 2], uvfaces[i, 2],
										#  faces[i, 1], uvfaces[i, 1],
										#  faces[i, 0], uvfaces[i, 0]
										faces[i, 0], uvfaces[i, 0],
										faces[i, 1], uvfaces[i, 1],
										faces[i, 2], uvfaces[i, 2]
								)
								)
						# write mtl
						with open(mtl_name, 'w') as f:
								f.write('newmtl %s\n' % material_name)
								s = 'map_Kd {}\n'.format(os.path.basename(texture_name)) # map to image
								f.write(s)

								if normal_map is not None:
										name, _ = os.path.splitext(obj_name)
										normal_name = f'{name}_normals.png'
										f.write(f'disp {normal_name}')
										# out_normal_map = normal_map / (np.linalg.norm(
										#     normal_map, axis=-1, keepdims=True) + 1e-9)
										# out_normal_map = (out_normal_map + 1) * 0.5

										cv2.imwrite(
												normal_name,
												# (out_normal_map * 255).astype(np.uint8)[:, :, ::-1]
												normal_map
										)
						skimage.io.imsave(texture_name, texture)

@click.command()
@click.option("--file", "filepath", type=click.Path(exists=True), required=True)
@click.option("--output", "output_dir", type=click.Path(), required=True)
def main(filepath: str, output_dir: str, plot: bool = False):
	# load image
	image = skimage.io.imread(filepath)
	image_w, image_h, _ = image.shape

	# compute mesh
	with mp.solutions.face_mesh.FaceMesh(
		static_image_mode=True,
		refine_landmarks=True,
		max_num_faces=1,
		min_detection_confidence=0.5) as face_mesh:
			results = face_mesh.process(image)

	assert len(results.multi_face_landmarks) == 1, "Only one face can be present in the image"
	face_landmarks = results.multi_face_landmarks[0]

	# covert to pixel coordinates
	keypoints = np.array([(image_w*point.x,image_h*point.y) for point in face_landmarks.landmark[:468]])

	if plot:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.imshow(image)
		ax.plot(keypoints[:, 0], keypoints[:, 1], '.b', markersize=2)
		plt.show()

	# load uv
	with open("./data/uv_map.json", "r") as file:
		uv_coords = json.load(file)
		uv_map = zip(uv_coords.get("u").values(), uv_coords.get("v").values())

	uv_w, uv_h = uv_shape = 512, 512
	uv_keypoints = np.array([(uv_w*x, uv_h*y) for x,y in uv_map]) # convert to pixel coordinates

	# transform uv to keypoints
	tform = PiecewiseAffineTransform()
	tform.estimate(uv_keypoints,keypoints)
	texture = warp(image, tform, output_shape=uv_shape)
	texture = (255*texture).astype(np.uint8)

	if plot:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.imshow(image)
		ax.plot(uv_keypoints[:, 0], uv_keypoints[:, 1], '.b', markersize=2)
		plt.show()

	keypoints3d = np.array([(point.x,point.y,point.z) for point in face_landmarks.landmark[:468]])

	face_template_filepath = "./data/canonical_face_model.obj"
	verts,uvcoords,faces,uv_faces = load_obj(face_template_filepath)

	vertices = normalize_keypoints(keypoints3d)
	obj_name =  "./results/obj_model.obj"
	write_obj(
		obj_name,
		vertices,
		faces,
		texture_name = "./results/texture.jpg",
		texture=texture,
		uvcoords=uvcoords,
		uvfaces=uv_faces,
	)

if __name__ == "__main__":
	main()