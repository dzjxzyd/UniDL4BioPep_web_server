# UniDL4BioPep_webserver
This is the webserver version of UniDL4BioPep at https://github.com/dzjxzyd/UniDL4BioPep.
Webserver is deployed on AWS AppRunner at https://nepc2pvmzy.us-east-1.awsapprunner.com/.

It is worthing noting, AWS AppRunner can only support keras 2 and tensorflow (2.12.1, based on our test, even do not support 2.15.0). because of the compability problem, the model here can only work with the keras2 tensorflow < 2.16, 

If your environment is keras 3 and tensorflow >= 2.16, you can download models at https://github.com/dzjxzyd/UniDL4BioPep_web_server_keras_3/tree/main. I test keras 3 with Apple M2 chips.

Running this server locally is very easy (following the steps below).

```
install python 3.8 (3.9 or 3.10 is also ok)
# download the repository and unzip the file
# install all the required packages (my running environment is MacOS Intel chip).
pip install requirements.txt
# go to the working directory
cd UniDL4BioPep_web_server
# run the server
python app.py

open the browser and go to this address    http://127.0.0.1:5000/

# the webserver is ready for usage

################################################################
########################result explaination#####################

sequence	Antihypertensive	Antihypertensive_probability	
QKTAP	        active	                  0.981061339	              
NNWNG	        non-active	          0.045210306          

# the above is an example results,
"Antihypertensive probability" this is an indicator for your, how confident the model with its prediction,

################################################################
######################## Additional notice #####################

1. In the Terminal Window, you will find the automatically output information, including
	name of the model; e.g., AHT_best_model.keras
        name of the scaler; e.g., AHT_minmax_scaler.pkl
	
2. if there is an error and in the Terminal said "can not allocate enough memory", that means the local machine can not process the embeddings,
	generally, the reason is the input sequence is toooooo long, maybe more than 8000 residues (Our local machine is very powerful, your local machine might occur the error with a shorter length)
	


```

