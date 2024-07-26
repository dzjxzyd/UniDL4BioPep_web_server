# UniDL4BioPep_webserver
This is the webserver version of UniDL4BioPep at https://github.com/dzjxzyd/UniDL4BioPep.
Webserver is based on AWS.

Running this server locally is very easy.

```
install python 3.8 (3.9 or 3.10 is also ok)
# download the repository and unzip the file
# install all the required packages.
pip install requirements.txt
# go to the working directory
cd UniDL4BioPep_web_server
# run the server
python app.py

open the browser and go to this address    http://127.0.0.1:5000/

# the webserver is ready for usage

################################################################
########################result explaination#####################
sequence	1_Antihypertensive	1_Antihypertensiveprotability	2_DPPIV	         2_DPPIVprotability
MAILGLGTDIVEIARIS	non-active	8.01560872787377E-07	        active	         0.9999231231223112

# the above is an example results,
"Antihypertensive protability" this is an indicator for your, how confident the model with its prediction,

for example, the 0.9999231231223112 means the model think the probability is 0.9999231231223112 for the DPPIV active prediction.
while the 8.01560872787377E-07 means the probability the model predict it is active (almost 0, so we finally generate the non-active output)

################################################################
######################## Additional notice #####################

1. In the Terminal Window, you will find the automatically output information, 
	include the sequence length of your input sequence, length of the seqeunce, and (1,320) (this 1,320 is the embedding matrix)
	
2. if there is an error and in the Terminal said "can not allocate enough memory", that means the local machine can not process the embeddings,
	generally, the reason is the input sequence is toooooo long, maybe more than 8000 residues (Our local machine is very powerful, you can only try to build this server on PC with A100 40GB GPU memory)
	


```


https://github.com/dzjxzyd/UniDL4BioPep_web_server/tree/main
