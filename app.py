import os
import numpy as np
import pandas as pd
from keras.models import load_model
from flask import Flask, request, render_template, send_from_directory
import joblib
from werkzeug.utils import secure_filename
# from focal_loss import BinaryFocalLoss
import torch
import esm
import collections
app = Flask(__name__)

# embeddings function
def esm_embeddings(peptide_sequence_list: list):
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval() 
    batch_labels, batch_strs, batch_tokens = batch_converter(peptide_sequence_list)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    token_representations = results["representations"][6]
    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
    # save dataset
    # sequence_representations is a list and each element is a tensor
    embeddings_results = collections.defaultdict(list)
    for i in range(len(sequence_representations)):
        # tensor can be transformed as numpy sequence_representations[0].numpy() or  sequence_representations[0].to_list
        each_seq_rep = sequence_representations[i].tolist()
        for each_element in each_seq_rep:
            embeddings_results[i].append(each_element)
    embeddings_results = pd.DataFrame(embeddings_results).T
    return embeddings_results

# collect the output
def assign_activity(predicted_class):
    import collections
    out_put = []
    for i in range(len(predicted_class)):
        if predicted_class[i] == 1:
            # out_put[int_features[i]].append(1)
            out_put.append('active')
        else:
            # out_put[int_features[i]].append(2)
            out_put.append('non-active')
    return out_put


def get_filetype(filename):
    return filename.rsplit('.', 1)[1].lower()

def model_selection(num: str):
    model = ''
    if num == '1':
        model = 'AHT_'
        activity_name = 'Antihypertensive'
    elif num == '2':
        model = 'DPPIV_'
        activity_name = 'DPPIV'
    elif num == '3':
        model = 'bitter_'
        activity_name = 'Bitter'
    elif num == '4':
        model = 'umami_'
        activity_name = 'Umami'
    elif num == '5':
        model = 'AMP_'
        activity_name = 'Antimicrobial'
    elif num == '6':
        model = 'Antimarial_alternative_'
        activity_name = 'Antimarial-alternative'
    elif num == '7':
        model = 'Antimalarial_main_'
        activity_name = 'Antimalarial-main'
    elif num == '8':
        model = 'QS_'
        activity_name = 'Quorum sensing'
    elif num == '9':
        model = 'ACP_alternative_'
        activity_name = 'Anticancer-alternative'
    elif num == '10':
        model = 'ACP_main_'
        activity_name = 'Anticancer-main'
    elif num == '11':
        model = 'anti_MRSA_'
        activity_name = 'Anti_MRSA'
    elif num == '12':
        model = 'TTCA_'
        activity_name = 'Tumor T-cell antigens'
    elif num == '13':
        model = 'BBP_'
        activity_name = 'Blood-Brain Barrier Penetrating Peptides'
    elif num == '14':
        model = 'APP_'
        activity_name = 'Anti-parasitic'
    elif num == '15':
        model = 'neuro_'
        activity_name = 'Neuropeptide'
    elif num == '16':
        model = 'AB_'
        activity_name = 'Antibacterial'
    elif num == '17':
        model = 'AF_'
        activity_name = 'Antifungal'
    elif num == '18':
        model = 'AV_'
        activity_name = 'Antiviral'
    elif num == '19':
        model = 'toxicity_'
        activity_name = 'Toxicity'
    elif num == '20':
        model = 'Antioxidant_'
        activity_name = 'Antioxidant'
    elif num == '21':
        model = 'Allergen_'
        activity_name = 'Allergen'
    elif num == '22':
        model = 'CPP_'
        activity_name = 'Cell Penetrating Peptide'
    else:
        raise ValueError("Model id should be one of the above 22 models")
    return model, activity_name


def text_fasta_reading(file_name):
    """
    A function for reading txt and fasta files
    """
    import collections
    # read txt file with sequence inside
    file_read = open(file_name, mode='r')
    file_content = []  # create a list for the fasta content temporaty storage
    for line in file_read:
        file_content.append(line.strip())  # extract all the information in the file and delete the /n in the file

    # build a list to collect all the sequence information
    sequence_name_collect = collections.defaultdict(list)
    for i in range(len(file_content)):
        if '>' in file_content[i]:  # check the symbol of the
            a = i+1
            seq_template = str()
            while a <len(file_content) and '>' not in file_content[a] and len(file_content[a])!= 0 :
                seq_template = seq_template + file_content[a]
                a=a+1
            sequence_name_collect[file_content[i]].append(seq_template)

    # transformed into the same style as the xlsx file loaded with pd.read_excel and sequence_list = dataset['sequence']
    sequence_name_collect = pd.DataFrame(sequence_name_collect).T
    sequence_list = sequence_name_collect[0]
    return sequence_list


def get_activity(model_name, sequence_list) -> list:
    # os.chdir('/Users/zhenjiaodu/Downloads/UniDL4BioPep_web_server-main_2')
    # model_name = '6_AMAP_main'
    # sequence_list=['QPFPQPQLPY','IPPYCTIAPV','SLQALRSMC']
    model_name_full = model_name + 'keras_2_best_model.keras'
    scaler_name = model_name + 'keras_2_minmax_scaler.pkl'
    print(model_name_full,scaler_name)
    model = load_model(model_name_full)
    scaler = joblib.load(scaler_name)
    # 因为这个list里又两个element我们需要第二个，所以我只需要把吧这个拿出来，然后split
    # 另外需要注意，这个地方，网页上输入的时候必须要是AAA,CCC,SAS, 这个格式，不同的sequence的区分只能使用逗号，其他的都不可以
    embeddings_results = pd.DataFrame()
    for seq in sequence_list:
        format_seq = [seq, seq]  # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list = []
        peptide_sequence_list.append(tuple_sequence)  # build a summarize list variable including all the sequence information
        one_seq_embeddings = esm_embeddings(peptide_sequence_list)  # conduct the embedding
        embeddings_results = pd.concat([embeddings_results,one_seq_embeddings])
        
    normalized_embeddings_results = scaler.transform(embeddings_results)  # normalized the embeddings
    # prediction
    predicted_probability = model.predict(normalized_embeddings_results, batch_size=1)
    predicted_class = np.argmax(predicted_probability, axis=1) # operating horizontally /// row-wise

    predicted_class_new = assign_activity(predicted_class) 
    return predicted_class_new, predicted_probability


# create an app object using the Flask class
@app.route('/')
def home():
    import os
    print(f"Current working directory: {os.getcwd()}")
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 每一个网页上的 输入的框，是一个单独的x，下面这个就是吧这个单独的信息变成一个list，每一个单独的就是一个str （也可以吧x变成int 如果想要的话）
    # int_features  = [str(x) for x in request.form.values()] # this command basically use extract all the input into a list
    # final_features = [np.array(int_features)]
    import os
    # print(os.listdir())
    # print(os.getcwd())
    int_features = [str(x) for x in request.form.values()]
    print(int_features)
    # we have two input in the website, one is the model type and other is the peptide sequences

    # choose scaler and model
    #    name = int_features[0]
    model_id = int_features[0]
    model_name, activity_name = model_selection(model_id)
    model_name_full = model_name + 'keras_2_best_model.keras'
    scaler_name = model_name + 'keras_2_minmax_scaler.pkl'
    
    print(model_name, activity_name)
    print(model_name_full,scaler_name)
    
    scaler = joblib.load(scaler_name)
    model = load_model(model_name_full)
    sequence_list = int_features[1].split(',')  # 因为这个list里又两个element我们需要第二个，所以我只需要把吧这个拿出来，然后split
    # 另外需要注意，这个地方，网页上输入的时候必须要是AAA,CCC,SAS, 这个格式，不同的sequence的区分只能使用逗号，其他的都不可以
    embeddings_results = pd.DataFrame()
    for seq in sequence_list:
        format_seq = [seq, seq]  # the setting is just following the input format setting in ESM model, [name,sequence]
        tuple_sequence = tuple(format_seq)
        peptide_sequence_list = []
        peptide_sequence_list.append(tuple_sequence)  # build a summarize list variable including all the sequence information
        one_seq_embeddings = esm_embeddings(peptide_sequence_list)  # conduct the embedding
        embeddings_results = pd.concat([embeddings_results,one_seq_embeddings])

    normalized_embeddings_results = scaler.transform(embeddings_results)  # normalized the embeddings
    # prediction
    predicted_probability = model.predict(normalized_embeddings_results, batch_size=1)
    predicted_class = np.argmax(predicted_probability, axis=1) # operating horizontally /// row-wise

    predicted_class_new = assign_activity(predicted_class) 

    final_output = []
    final_output.append(activity_name + ': ')
    for i in range(len(sequence_list)):
        # print()
        print(predicted_class_new[i])
        temp_output=sequence_list[i]+': '+ predicted_class_new[i] + ' '+ str(round(predicted_probability[i, 1],4)) + ';' 
        final_output.append(temp_output)

    return render_template('index.html',
                           prediction_text="Prediction results of input sequences{}".format(final_output))


@app.route('/pred_with_file', methods=['POST'])
def pred_with_file():
    # delete existing files that are in the 'input' folder
    dir = 'input'
    for f in os.listdir(os.path.join(os.getcwd(), dir)):
        os.remove(os.path.join(dir, f))
    # 每一个网页上的 输入的框，是一个单独的x，下面这个就是吧这个单独的信息变成一个list，每一个单独的就是一个str （也可以吧x变成int 如果想要的话）
    # int_features  = [str(x) for x in request.form.values()] # this command basically use extract all the input into a list
    # final_features = [np.array(int_features)]
    features = request.form  # .values()
    # we have two input in the website, one is the model type and other is the peptide sequences
    # choose scaler and model
    #    name = int_features[0]
    models = features.getlist("Model_selection")
    if len(models) == 0:  # didn't check any model
        return home()
    file = request.files["Peptide_sequences"]
    filename = secure_filename(file.filename)
    filetype = get_filetype(filename)
    save_location = os.path.join('input', filename)
    file.save(save_location)
    sequence_list = []
    if filetype == 'xls' or filetype == 'xlsx':
        df = pd.read_excel(save_location, header=0)
        sequence_list = df["sequence"].tolist()
    if filetype == 'txt' or filetype == 'fasta':
        sequence_list = text_fasta_reading(save_location)

    if len(sequence_list) == 0:
        return home()

    report = {"sequence": sequence_list}
    print(models)
    for model in models:
        model_name, activity_name = model_selection(model)
        activities, probability = get_activity(model_name, sequence_list)
        report[activity_name] = activities
        probability_column_model = activity_name + '_probability'
        report[probability_column_model] = probability[:,1]

    report_df = pd.DataFrame(report)
    save_result_path = os.path.join('input', "report.xlsx")
    report_df.to_excel(save_result_path)
    send_from_directory("input", "report.xlsx")

    return send_from_directory("input", "report.xlsx")


if __name__ == '__main__':
    app.run()
