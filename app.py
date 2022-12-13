import streamlit as st
import pandas as pd

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

def f_side():
    #st.sidebar.image("https://as2.ftcdn.net/v2/jpg/05/23/82/65/1000_F_523826590_gkVPPuLEG4aijfh8bJviQCJH6oSERwlb.jpg", width=150)
    st.sidebar.subheader('Ficha do paciente')
    
def f_dados_paciente():      
    nome_pac = st.sidebar.text_input('Nome:', key='nome')
    idade_pac = st.sidebar.text_input('Idade:', key='idade')     
    sexo_pac = st.sidebar.radio("Sexo", ["Masculino", "Feminino"],
        key="sexo", label_visibility='visible', disabled=False, horizontal=True)
    raca_pac = st.sidebar.selectbox("Raça",("Selecione", "Branca", "Preta", "Amarela","Parda","Indígena","Ignorado"),
        key="raca")
    # zona_pac = st.sidebar.selectbox("Zona",("Selecione", "Urbana", "Rural", "Periurbana","Ignorado"),
    #     key="zona")
    vac_pac = st.sidebar.checkbox('Recebeu vacina?', key='vacina')
    #st.sidebar.write('Fatores de risco:')
    st.sidebar.markdown(f'<h4 style="color: #ff0000;"><strong>Fatores de Risco:</strong></h4>', unsafe_allow_html=True)
    obes_pac = st.sidebar.checkbox('Obesidade', key="obesidade")
    asma_pac = st.sidebar.checkbox('Asma', key="asma")
    diab_pac = st.sidebar.checkbox('Diabetes', key="diabetes")

    dir_pac = {'Nome': nome_pac, 'Idade': idade_pac, 'Sexo': sexo_pac, 'Raça': raca_pac,
    'Vacina': vac_pac, 'Obesidade': obes_pac, 'Asma': asma_pac, 'Diabetes': diab_pac}
    features=pd.DataFrame(dir_pac, index=[0])
    return features

def f_modelo():
    df = pd.read_csv('dados_agrupados.csv', delimiter=';', quotechar='"')
    return df

x = f_side()
x = f_dados_paciente()

col1, col2 = st.columns(2)
with col1:
    col1.header("""
    Projeto Data Science   Covid Longa\n
    App que utiliza machine learning para prever possível covid longa dos pacientes
    """)

    st.subheader('Instruções:')
    st.write("1- Preencher a ficha do paciente ao lado esquerdo desta tela")
    st.write('2- Clique no botão <Confirmar> abaixo')
    st.write('3- Será apresentado o resultado da avaliação; para incluir os dados de outro paciente, clique no botão <Nova Ficha>')


    def clear_text():
        st.session_state["nome"] = ""
        st.session_state["idade"] = ""
        st.session_state["vacina"] = False
        st.session_state["obesidade"] = False
        st.session_state["raca"] = "Selecione"
        #st.session_state["zona"] = "Selecione"
        st.session_state["asma"] = False
        st.session_state["diabetes"] = False

    #st.write(x)

    # if st.button("Confirmar"):
    #     st.balloons()
    #     st.success('Sem covid longa   :)')
    #     st.warning('Covid longa   :(')
    #     df = f_modelo()
  

with col2:
    #st.image("https://as1.ftcdn.net/jpg/03/07/43/75/240_F_307437510_x6kug0WyeBJQjzhjVs3jTbIQkpJBDPP1.jpg", width=300)
    col2.markdown("""<p><img style="float: right;" src="https://as1.ftcdn.net/jpg/03/07/43/75/240_F_307437510_x6kug0WyeBJQjzhjVs3jTbIQkpJBDPP1.jpg" alt="" width="240" height="240" /></p>""", unsafe_allow_html=True)
    

col3, col4, col5 = st.columns(3)
with col3:
    if st.button("Confirmar"):
        st.balloons()
        st.success('Sem covid longa   :)')
        st.warning('Covid longa   :(')
        df = f_modelo()

with col4:
    if st.button("Imprimir"):
        #arquivo = open('ficha.pdf', 'r')
        st.balloons()

with col5:
    st.button("Nova ficha", on_click=clear_text)

    