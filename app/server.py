import os
import sys
import cv2
import math
import typing
import aiohttp
import asyncio
import uvicorn
import numpy as np
from io import BytesIO
import face_recognition
from pathlib import Path
from PIL.ExifTags import TAGS
from PIL import Image, ImageFile, ImageFilter, ImageEnhance
from starlette.background import BackgroundTask
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse

export_file_url  = 'https://live.staticflickr.com/65535/49845513266_92f41da548_o_d.png'
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
	if os.path.exists("app/newmask.png"):
		os.remove("app/newmask.png")
	else:
		print ("The file does not exist")
	return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST','GET'])
async def analyze(request):
	
	img_data = await request.form()
	# print (img_data)
	# print(img_data['emotion'])
	img_bytes = await (img_data['file'].read())
	img = Image.open(BytesIO(img_bytes))
	img_exif = img.getexif()
	if img_exif:
		for key,value in img._getexif().items():
			if TAGS.get(key) == 'Orientation':
				orientation = value
		if orientation == 1:
			img 
		if orientation == 3:
			img = img.rotate(180)
		if orientation == 6:
			img = img.rotate(270)
		if orientation == 8:
			img= img.rotate(90)
	else:
		print("image has no ExifTags")

	max_size=512

	if img.height > max_size or img.width > max_size:
		# if width > height:
		if img.width > img.height:
			desired_width = max_size
			desired_height = img.height / (img.width/max_size)
				
		# if height > width:
		elif img.height > img.width:
			desired_height = max_size
			desired_width = img.width / (img.height/max_size)
				
		else:
			desired_height = max_size
			desired_width = max_size
				
		# convert back to integer
		desired_height = int(desired_height)
		desired_width = int(desired_width)
				
		img = img.resize((desired_width, desired_height))

	else:
		print ('img reize not required')

	if os.path.exists("app/newmask.png"):
		os.remove("app/newmask.png")
	else:
		print("The file does not exist")
	
	def cli(pic_path ,save_pic_path ):
		mask_path = BLACK_IMAGE_PATH
		FaceMasker(pic_path, mask_path, True, 'hog',save_pic_path).mask()
	
	
	
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
						print ("no")
						break
				if skip:
					continue

				# mask face
				found_face = True
				self._mask_face(face_landmark)
				# self.face_image=self.face_image.filter(ImageFilter.SMOOTH)
				new_face_path = path/'newmask.png' 
				self._face_img.save(new_face_path)

				#filter operation(1)
				def contrast(source_name, result_name, coefficient):
					source = Image.open(source_name)
					result = Image.new('RGB', source.size)

					avg = 0
					for x in range(source.size[0]):
						for y in range(source.size[1]):
							r, g, b = source.getpixel((x, y))
							avg += r * 0.299 + g * 0.587 + b * 0.114
					avg /= source.size[0] * source.size[1]

					palette = []
					for i in range(256):
						temp = int(avg + coefficient * (i - avg))
						if temp < 0:
							temp = 0
						elif temp > 255:
							temp = 255
						palette.append(temp)

					for x in range(source.size[0]):
						for y in range(source.size[1]):
							r, g, b = source.getpixel((x, y))
							result.putpixel((x, y), (palette[r], palette[g], palette[b]))

					result.save(path/'newmask.png', "PNG")

				contrast(path/'newmask.png',path/'newmask.png',1.5)

				# im.save(path/'newmask.png',"PNG")

				# self._save()

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

		# def _save(self):
		# 	new_face_path = path/'newmask.png'
		# 	self._face_img.save(new_face_path)
		# 	print(f'Save to {new_face_path}')

		@staticmethod
		def get_distance_from_point_to_line(point, line_point1, line_point2):
			distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
				(line_point1[0] - line_point2[0]) * point[1] +
				(line_point2[0] - line_point1[0]) * line_point1[1] +
				(line_point1[1] - line_point2[1]) * line_point1[0]) / \
			np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
				(line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
			return int(distance)






	if __name__ == '__main__':
		save_imgpath = path
		cli(img,save_imgpath)
	return FileResponse('app/newmask.png',media_type='image/png')


@app.route("/download",methods=['POST','GET'])
async def  download(request):
	# task = BackgroundTask(rem)
	return FileResponse("app/newmask.png",media_type='image/png')




if __name__ == '__main__':
	if 'serve' in sys.argv:
		uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
