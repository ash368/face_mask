import aiohttp
import asyncio
import uvicorn
import os
import sys
import argparse
import numpy as np
import cv2
import math
from PIL import Image, ImageFile
from pathlib import Path
import typing
import warnings
from typing import Union



from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse

export_file_url  = 'https://github.com/ash368/face_mask/raw/master/masks/black-mask.png'
export_file_name = 'black-mask.png'

path = Path(__file__).parent


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
	if dest.exists(): return
	async with aiohttp.ClientSession() as session:
		async with session.get(url) as response:
			data = await response.read()
			with open(dest, 'wb') as f:
				f.write(data)




async def setup_mask():
	await download_file(export_file_url, path / export_file_name)
	BLACK_IMAGE_PATH = path/export_file_name
	return BLACK_IMAGE_PATH
	
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_mask())]
BLACK_IMAGE_PATH = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
async def homepage(request):
	html_file = path / 'view' / 'index.html'
	return HTMLResponse(html_file.open().read())




@app.route('/analyze', methods=['POST'])
async def analyze(request):
	
	img_data = await request.form()
	img_bytes = await (img_data['file'].read())
	img = Image.open(BytesIO(img_bytes))



	# img = img.save('newer.jpg')
	
	
	
	class FaceMasker:
		KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

		def __init__(self, face_path, mask_path, show=False, model='hog',save_path = ''):
			self.face_path = face_path
			self.mask_path = mask_path
			self.save_path = save_path
			self.show = show
			self.model = model
			self._face_img: ImageFile = None
			self._mask_img: ImageFile = None

		def mask(self):
		
			import face_recognition

			# face_image_np = face_recognition.load_image_file(self.face_path)
			face_image_np = np.array(img)
			face_locations = face_recognition.face_locations(face_image_np, model=self.model)
			face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
			self._face_img = Image.fromarray(face_image_np)
			self._mask_img = Image.open(self.mask_path)

			found_face = False
			for face_landmark in face_landmarks:
				# check whether facial features meet requirement
				skip = False
				for facial_feature in self.KEY_FACIAL_FEATURES:
					if facial_feature not in face_landmark:
						skip = True
						break
				if skip:
					continue

				# mask face
				found_face = True
				self._mask_face(face_landmark)

				if self.show:
					self._face_img.show()
				# save
				self._save()

		def _mask_face(self, face_landmark: dict):
			nose_bridge = face_landmark['nose_bridge']
			nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
			nose_v = np.array(nose_point)

			chin = face_landmark['chin']
			chin_len = len(chin)
			chin_bottom_point = chin[chin_len // 2]
			chin_bottom_v = np.array(chin_bottom_point)
			chin_left_point = chin[chin_len // 8]
			chin_right_point = chin[chin_len * 7 // 8]

			# split mask and resize
			width = self._mask_img.width
			height = self._mask_img.height
			width_ratio = 1.2
			new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

			# left
			mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
			mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
			mask_left_width = int(mask_left_width * width_ratio)
			mask_left_img = mask_left_img.resize((mask_left_width, new_height))

			# right
			mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
			mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
			mask_right_width = int(mask_right_width * width_ratio)
			mask_right_img = mask_right_img.resize((mask_right_width, new_height))

			# merge mask
			size = (mask_left_img.width + mask_right_img.width, new_height)
			mask_img = Image.new('RGBA', size)
			mask_img.paste(mask_left_img, (0, 0), mask_left_img)
			mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

			# rotate mask
			angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
			rotated_mask_img = mask_img.rotate(angle, expand=True)

			# calculate mask location
			center_x = (nose_point[0] + chin_bottom_point[0]) // 2
			center_y = (nose_point[1] + chin_bottom_point[1]) // 2

			offset = mask_img.width // 2 - mask_left_img.width
			radian = angle * np.pi / 180
			box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
			box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

			# add mask
			self._face_img.paste(mask_img, (box_x, box_y), mask_img)

		def _save(self):
			# path_splits = os.path.splitext(self.face_path)
			# new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
			
			

			new_face_path = path/'newmask.png'
			self._face_img.save(new_face_path)
			print(f'Save to {new_face_path}')

		@staticmethod
		def get_distance_from_point_to_line(point, line_point1, line_point2):
			distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
				(line_point1[0] - line_point2[0]) * point[1] +
				(line_point2[0] - line_point1[0]) * line_point1[1] +
				(line_point1[1] - line_point2[1]) * line_point1[0]) / \
			np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
				(line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
			return int(distance)

	def cli(pic_path ,save_pic_path ):
		mask_path = BLACK_IMAGE_PATH
		FaceMasker(pic_path, mask_path, True, 'hog',save_pic_path).mask()




	if __name__ == '__main__':
		# imgpath = os.path.join(root, name)
		save_imgpath = path
		cli(img,save_imgpath)

	# return StreamingResponse('app/newmask.png', media_type="image/png")
	return FileResponse('app/newmask.png',media_type='image/png')

if __name__ == '__main__':
	if 'serve' in sys.argv:
		uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
