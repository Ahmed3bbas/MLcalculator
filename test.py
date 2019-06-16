import unittest
from Calculator import predict
import cv2
import glob

class TestCaculator(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		
		cls.eq = ['3*2+1',
				'3+1*5',
				'Operation Failed',
				'3+*4']

		cls.result = [7,
					  8,
					  "The Equation is Not Correct",
					  "The Equation is Not Correct"]

		paths = r"D:\3 year of comuter engineering\second term\SW2\DigitRecognition-master\src\Calculator\img"
		images = glob.glob(paths + r'\*.jpg')
		
		cls.tested_img = []
		cls.actual_otput_img = []

		for filename in images:

			img = cv2.imread(filename)
			if filename.find("input") > 1:
				cls.tested_img.append(img)
			elif filename.find("predict") > 1:
				cls.actual_otput_img.append(img)
	
	@classmethod
	def tearDownClass(cls):
		pass

	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_predict(self):
		i = 0

		for _input in self.tested_img:

			im, equation, res = predict(_input)
			im = cv2.resize(im,(350,360))

			self.assertEqual(im.all(), self.actual_otput_img[i].all())
			self.assertEqual(equation, self.eq[i])
			self.assertEqual(res, self.result[i])

			i+=1
		#self.assertRaises(Exception,predict,self.tested_img[2])


if __name__ == '__main__':
    unittest.main()