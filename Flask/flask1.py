


from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array

app = Flask(__name__)

# dic = {0 : 'red_soil', 1 : 'black_soil', 2: 'clay_soil',3:'alluvial_soil'}

model = load_model('model2_MobileNetV2.h5')
label=['Alluvial', 'black', 'Clay', 'Red']
model.make_predict_function()

def predict_label(img_path):
	pred_1 = []
	image_1=cv2.imread(img_path)
	image_1 = cv2.resize(image_1, (128, 128))
	image_1 = img_to_array(image_1)
	pred_1.append(image_1)
	pred_1 = np.array(pred_1, dtype="float32") / 255.0
	image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
	#cv2.imshow("prediction", image_1)
	result_2=model.predict(pred_1)
	si_la = label[np.argmax(result_2)]
	
	return si_la

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("neww.html")

@app.route("/about")
def about_page():
	return ""

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		print(img_path)
		img.save(img_path)
		

		try:
			p = predict_label(img_path)
			if p== "black":
				a=("SOIL TYPE : Black soil :: CROPS :sugarcane,wheat,Cotton,soybeans,Pigeon peas,Sunflower, Maize,Chickpeas")
			elif p== "Alluvial":
				a=("SOIL TYPE: Alluvial soil :: CROPS :Paddy rice,Coconut palms,Banana,tomatoes, beans, cucumber,pepper, cardamom, turmeric, and ginger,mango, jackfruit, pineapple, guava, and papaya,Rubber trees,Cashew nut trees")
			elif p== "Clay":
				a=("SOIL TYPE :Clay soil :: CROPS :Tapioca,Turmeric ,Ginger,Yam,Sugarcane,Banana")
			else:
				a=("SOIL TYPE :Red soil :: CROPS : Rubber trees,Black pepper,Coffee,Cardamom,Turmeric,Pineapple, Cloves,Banana" )
			
		except:
			p = "An error occurred while processing the image."

		

	return render_template("neww.html", prediction = a, img_path = img_path)





# In[4]:


if __name__ =='__main__':
	app.debug = True
	app.run(debug = True)


# In[ ]:




