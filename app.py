pip install -U scikit-learn

import streamlit as st
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,plot_confusion_matrix
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression,f_classif
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression


hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

def f_side():
    #st.sidebar.image("https://as2.ftcdn.net/v2/jpg/05/23/82/65/1000_F_523826590_gkVPPuLEG4aijfh8bJviQCJH6oSERwlb.jpg", width=150)
    st.sidebar.write('SysCovid - versão 1.0')
    st.sidebar.subheader('Ficha do paciente')
    
def f_dados_paciente():      
    nome_pac = st.sidebar.text_input('Nome:', key='nome')
    idade_pac = st.sidebar.text_input('Idade:', key='idade')     
    sexo_pac = st.sidebar.radio("Sexo", ["Masculino", "Feminino"],
        key="sexo", label_visibility='visible', disabled=False, horizontal=True)
    raca_pac = st.sidebar.selectbox("Raça",("Selecione", "Branca", "Preta", "Amarela","Parda","Indígena","Ignorado"),
        key="raca")

    vac_pac = st.sidebar.checkbox('Recebeu vacina?', key='vacina')
    vac_nosocomial = st.sidebar.checkbox('Infecção adquirida no hospital?', key='nosocomial')
    st.sidebar.markdown(f'<h4 style="color: #ff0000;"><strong>Fatores de Risco:</strong></h4>', unsafe_allow_html=True)
    cardio_pac = st.sidebar.checkbox('Doença Cardiovascular Crônica', key="cardiopatia")
    hemato_pac = st.sidebar.checkbox('Doença Hematológica Crônica', key="hemato")  
    neuro_pac = st.sidebar.checkbox('Doença Neurológica Crônica', key='neuro')
    renal_pac = st.sidebar.checkbox('Doença Renal Crônica', key='renal')    
    diab_pac = st.sidebar.checkbox('Diabetes', key="diabetes")
    dispneia_pac = st.sidebar.checkbox('Dispneia', key="dispneia")
    imuno_pac = st.sidebar.checkbox('Imunodeficiência', key='imuno')
    obes_pac = st.sidebar.checkbox('Obesidade', key="obesidade")
    pneumo_pac = st.sidebar.checkbox('Outra Pneumatopatia Crônica', key='pneumo')
    sdown_pac = st.sidebar.checkbox('Síndrome de Down', key='sdown')

    dic_pac = {'Nome': nome_pac, 'Idade': idade_pac, 'Sexo': sexo_pac, 'Raça': raca_pac,
    'Vacina': vac_pac, 'Nosocomial': vac_nosocomial,'Cardiopatia': cardio_pac,
    'Diabetes': diab_pac,'Dispneia':dispneia_pac,'Hematologica':hemato_pac, 'Imunologica':imuno_pac,
    'Neurologica': neuro_pac,'Obesidade': obes_pac,'Pneumatopatia':pneumo_pac,'Renal':renal_pac,'SDown':sdown_pac}

    if raca_pac == 'Branca':
        racaBranca=1
        racaParda=0
        racaOutras=0
    elif raca_pac == 'Parda':
        racaBranca=0
        racaParda=1
        racaOutras=0
    else:
        racaBranca=0
        racaParda=0
        racaOutras=1
    
    if idade_pac < 20:
        jovem=1
        adulto=0
        idoso=0
    elif idade_pac < 60:
        jovem=0
        adulto=1
        idoso=0
    else:
        jovem=0
        adulto=0
        idoso=1

    if sexo_pac == 'Masculino':
        masc=1
        fem=0
    else:
        masc=0
        fem=1

    dic_pac={'Branca': racaBranca, 'Parda': racaParda, 'Outras': racaOutras, 'Jovem': jovem, 'Adulto': adulto,
    'Idoso': idoso, 'Fem': fem, 'Masc': masc, 'vacina': 0, 'nosocomial': 0, 'dispneia': 0, 'cardiopatia': 0, 'hematologica': 0, 'sindrome_down': 0, 'diabetes': 1, 'neurologica': 0, 'pneumopatia': 0, 'imunodepressao': 0, 'renal': 0, 'obesidade': 1}
    
    features=pd.DataFrame(dic_pac, index=[0])
    return features

def f_modelo():
    df = pd.read_csv('dados_covid.csv', delimiter=';', quotechar='"')
    # Selected Columns
    features=['Branca','Parda','Outras','Jovem','Adulto','Idoso','Fem','Masc','vacina','nosocomial','dispneia','cardiopatia','hematologica','sindrome_down','diabetes','neurologica','pneumopatia','imunodepressao','renal','obesidade']
    target='covid_longa'
    # X & Y
    X=df[features]
    Y=df[target]
    # Data split for training and testing
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
    # Model Initialization
    model=LogisticRegression(C=0.118, class_weight={}, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=1, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
    model.fit(X_train,Y_train)
    y_pred=model.predict(X_test)
    return model
    # teste = {'POSCOMP': 65, 'Inglês': 6, 'Artigos publicados': 2}
    # dft = pd.DataFrame(data = teste,index=[0])
    # print(dft)
    # resultado = model.predict(dft)

f_side()
model=f_modelo()
df = f_dados_paciente()

col1, col2 = st.columns(2)
with col1:
    col1.header("""
    Projeto Data Science   Covid Longa\n
    App que utiliza machine learning para prever possível covid longa dos pacientes
    """)

    st.subheader('Instruções:')
    
# st.write("1- Preencher a ficha do paciente ao lado esquerdo desta tela")
# st.write('2- Clique no botão <Confirmar> para ver o resultado da avaliação')
# st.write('3- Clique no botão <Imprimir> para imprimir a ficha do paciente')
# st.write('4- Clique no botão <Nova ficha> para finalizar esta ficha e abrir nova')

    def clear_text():
        st.session_state["nome"] = ""
        st.session_state["idade"] = ""
        st.session_state["raca"] = "Selecione"
        st.session_state["vacina"] = False        
        st.session_state["nosocomial"] = False
        st.session_state["cardiopatia"] = False
        st.session_state["hemato"] = False
        st.session_state["diabetes"] = False
        st.session_state["dispneia"] = False
        st.session_state["imuno"] = False
        st.session_state["neuro"] = False
        st.session_state["pneumo"] = False
        st.session_state["obesidade"] = False
        st.session_state["renal"] = False
        st.session_state["sdown"] = False

    #st.write(df)

    # if st.button("Confirmar"):
    #     st.balloons()
    #     st.success('Sem covid longa   :)')
    #     st.warning('Covid longa   :(')
    #     df = f_modelo()
  

with col2:
    #st.image("https://as1.ftcdn.net/jpg/03/07/43/75/240_F_307437510_x6kug0WyeBJQjzhjVs3jTbIQkpJBDPP1.jpg", width=300)
    col2.markdown("""<p><img style="float: right;" src="https://as1.ftcdn.net/jpg/03/07/43/75/240_F_307437510_x6kug0WyeBJQjzhjVs3jTbIQkpJBDPP1.jpg" alt="" width="240" height="240" /></p>""", unsafe_allow_html=True)
    
st.write("1- Preencher a ficha do paciente ao lado esquerdo desta tela")
st.write('2- Clique no botão <Confirmar> para obter o resultado da avaliação')
#st.write('3- Clique no botão <Imprimir> para imprimir a ficha do paciente')
st.write('3- Clique no botão <Nova ficha> para finalizar e iniciar uma nova ficha do paciente')

col3, col5 = st.columns(2)
with col3:
    if st.button("Confirmar"):
        if st.session_state["nome"]=='':
            st.warning('Por favor, informe o nome do paciente', icon="⚠️")
        elif st.session_state["idade"]=='':
            st.warning('Por favor, informe a idade do paciente', icon="⚠️")    
        elif st.session_state["raca"]=='Selecione':
            st.warning('Por favor, informe a raça do paciente', icon="⚠️")
        else:
            st.balloons()
            st.success('Sem covid longa   :)')
            st.warning('Covid longa   :(')
            #df = f_modelo()
            #teste = {'POSCOMP': 65, 'Inglês': 6, 'Artigos publicados': 2}
            #dft = pd.DataFrame(data = teste,index=[0])
            print(df)
            resultado = model.predict(df)
            print(resultado)

# with col4:
#     if st.button("Imprimir"):
#         path = 'https://github.com/claudio-mas/app_covid_longa/blob/e378bdb11fbc03a3f96353841ff40eb1e2162094/ficha.pdf'
#         #subprocess.Popen([path], shell=True)
#         #webbrowser.open_new(path)
#         os.system(path)
#         st.balloons()

with col5:
    st.button("Nova ficha", on_click=clear_text)

    