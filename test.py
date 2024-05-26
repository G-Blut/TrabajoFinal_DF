import streamlit as st
import matplotlib.pyplot as plt
from logica_FIS_gestion_trafico import QV, TE, HD, CM, TS, ts_sim  

# Define las funciones para cada funcionalidad
def pantalla_principal():
    st.title('Sistema de Control de Semáforo')
    # Botones para cada pantalla
    if st.button('Calcular Tiempo de Semáforo'):
        calcular_tiempo_semaforo()
    if st.button('Funcionalidad 1'):
        funcionalidad_1()
    if st.button('Funcionalidad 2'):
        funcionalidad_2()
    if st.button('Funcionalidad 3'):
        funcionalidad_3()
    if st.button('Funcionalidad 4'):
        funcionalidad_4()
    if st.button('Funcionalidad 5'):
        funcionalidad_5()
    if st.button('Funcionalidad 6'):
        funcionalidad_6()
    if st.button('Funcionalidad 7'):
        funcionalidad_7()

def calcular_tiempo_semaforo():
    st.title('Resultado')
    # Agrega el código para calcular el tiempo de semáforo y mostrar el resultado
    with st.sidebar:
        st.title("Antecedentes")
        # Crear sliders para las entradas del sistema
        st.header("Cantidad de vehiculos")
        qv_input = st.slider("", 0, 10, 5)  # El valor por defecto es 5
        st.header('Tiempo de Espera')
        te_input = st.slider('', 0, 10, 2)  # El valor por defecto es 2
        st.header("Hora del Día")
        hd_input = st.slider('', 0, 10, 4)  # El valor por defecto es 4
        st.header("Condiciones Meteorológicas")
        cm_input = st.slider('', 0, 10, 10)  # El valor por defecto es 10

        # Botón para realizar la simulación
        if st.button('Calcular Tiempo de Semáforo'):
            ts_sim.input['Cantidad de vehiculos'] = qv_input
            ts_sim.input['Tiempo de espera'] = te_input
            ts_sim.input['Hora del dia'] = hd_input
            ts_sim.input['Condiciones meteorologicas'] = cm_input

            # Computar la salida
            ts_sim.compute()

    # Mostrar el resultado
    output_ts = ts_sim.output['Tiempo de semaforo']
    st.write(f'El tiempo recomendado para el semáforo es: {output_ts} segundos.')

    # No mostrar warning deprecation
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Mostrar gráfico resultado
    st.pyplot(TS.view(sim=ts_sim))

def funcionalidad_1():
    st.title('Funcionalidad 1')
    # Agrega aquí el código para la funcionalidad 1

def funcionalidad_2():
    st.title('Funcionalidad 2')
    # Agrega aquí el código para la funcionalidad 2

def funcionalidad_3():
    st.title('Funcionalidad 3')
    # Agrega aquí el código para la funcionalidad 3

def funcionalidad_4():
    st.title('Funcionalidad 4')
    # Agrega aquí el código para la funcionalidad 4

def funcionalidad_5():
    st.title('Funcionalidad 5')
    # Agrega aquí el código para la funcionalidad 5

def funcionalidad_6():
    st.title('Funcionalidad 6')
    # Agrega aquí el código para la funcionalidad 6

def funcionalidad_7():
    st.title('Funcionalidad 7')
    # Agrega aquí el código para la funcionalidad 7

# Maneja la navegación
def main():
    estado = {'pantalla_actual': 'principal'}

    if estado['pantalla_actual'] == 'principal':
        pantalla_principal()
    # Puedes agregar más condiciones para otras funcionalidades si es necesario

if __name__ == '__main__':
    main()
