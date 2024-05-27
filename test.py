import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd
from pyds import MassFunction

# Definir las variables de entrada y salida difusas
demanda_predicha = ctrl.Antecedent(np.arange(0, 1001, 1), 'demanda_predicha')
impacto_promocion = ctrl.Antecedent(np.arange(0, 1001, 1), 'impacto_promocion')
inventario_rfid = ctrl.Antecedent(np.arange(0, 1001, 1), 'inventario_rfid')

valoracion_situacion = ctrl.Consequent(np.arange(0, 11, 1), 'valoracion_situacion')

# Definir las funciones de membresía para cada variable
demanda_predicha['baja'] = fuzz.trimf(demanda_predicha.universe, [0, 0, 500])
demanda_predicha['media'] = fuzz.trimf(demanda_predicha.universe, [250, 500, 750])
demanda_predicha['alta'] = fuzz.trimf(demanda_predicha.universe, [500, 1000, 1000])

impacto_promocion['bajo'] = fuzz.trimf(impacto_promocion.universe, [0, 0, 500])
impacto_promocion['medio'] = fuzz.trimf(impacto_promocion.universe, [250, 500, 750])
impacto_promocion['alto'] = fuzz.trimf(impacto_promocion.universe, [500, 1000, 1000])

inventario_rfid['bajo'] = fuzz.trimf(inventario_rfid.universe, [0, 0, 500])
inventario_rfid['medio'] = fuzz.trimf(inventario_rfid.universe, [250, 500, 750])
inventario_rfid['alto'] = fuzz.trimf(inventario_rfid.universe, [500, 1000, 1000])

valoracion_situacion['baja'] = fuzz.trimf(valoracion_situacion.universe, [0, 0, 5])
valoracion_situacion['media'] = fuzz.trimf(valoracion_situacion.universe, [2, 5, 8])
valoracion_situacion['alta'] = fuzz.trimf(valoracion_situacion.universe, [5, 10, 10])

# Crear las reglas difusas
rules = []

for demanda_level in ['baja', 'media', 'alta']:
    for impacto_level in ['bajo', 'medio', 'alto']:
        for inventario_level in ['bajo', 'medio', 'alto']:
            if demanda_level == 'alta' or impacto_level == 'alto' or inventario_level == 'bajo':
                valoracion_output = 'alta'
            elif demanda_level == 'media' and impacto_level == 'medio' and inventario_level == 'medio':
                valoracion_output = 'media'
            else:
                valoracion_output = 'baja'
            
            rule = ctrl.Rule(demanda_predicha[demanda_level] & impacto_promocion[impacto_level] & inventario_rfid[inventario_level], valoracion_situacion[valoracion_output])
            rules.append(rule)

valoracion_ctrl = ctrl.ControlSystem(rules)
valoracion_sim = ctrl.ControlSystemSimulation(valoracion_ctrl)

def funcionalidad_1():
    st.title('Dempster-Shafer Nivel 2')

    # Cargar las predicciones
    predicciones_demanda = pd.read_csv('predicciones_demanda_clasificador.csv')
    predicciones_impacto_promocion = pd.read_csv('predicciones_inventario_necesario.csv')
    predicciones_inventario_rfid = pd.read_csv('predicciones_inventario_rfid.csv')

    valoraciones = []

    for i in range(len(predicciones_demanda)):
        valoracion_sim.input['demanda_predicha'] = predicciones_demanda['Demanda_Predicha'][i]
        valoracion_sim.input['impacto_promocion'] = predicciones_impacto_promocion['Inventario_Necesario'][i]
        valoracion_sim.input['inventario_rfid'] = predicciones_inventario_rfid['Inventario_RFID'][i]
        valoracion_sim.compute()
        valoraciones.append(valoracion_sim.output['valoracion_situacion'])

    valoraciones_df = pd.DataFrame({'Valoracion_Situacion': valoraciones})
    valoraciones_df.to_csv('valoraciones_situacion.csv', index=False)
    st.write("Valoraciones de la situación guardadas en 'valoraciones_situacion.csv'")

    # Visualización
    fig, ax = plt.subplots()
    ax.plot(valoraciones_df.index, valoraciones_df['Valoracion_Situacion'], marker='o', linestyle='-')
    ax.set_xlabel('Índice')
    ax.set_ylabel('Valoración de la Situación')
    ax.set_title('Valoraciones de la Situación')
    st.pyplot(fig)

    # Dempster-Shafer
    valoraciones_df = pd.read_csv('valoraciones_situacion.csv')
    mass_functions = []

    for i in range(len(valoraciones_df)):
        valoracion = valoraciones_df['Valoracion_Situacion'][i]
        
        if valoracion <= 3.33:
            mass_function = MassFunction({'baja': 1.0})
        elif valoracion <= 6.66:
            mass_function = MassFunction({'media': 1.0})
        else:
            mass_function = MassFunction({'alta': 1.0})
        
        mass_functions.append(mass_function)

    combined_mass = mass_functions[0]
    for mass_function in mass_functions[1:]:
        combined_mass = combined_mass & mass_function

    belief = combined_mass.bel({'baja', 'media', 'alta'})
    plausibility = combined_mass.pl({'baja', 'media', 'alta'})
    pignistic = combined_mass.pignistic()

    st.write(f"Creencia: {belief}")
    st.write(f"Plausibilidad: {plausibility}")
    st.write(f"Probabilidad Pignística: {pignistic}")

    resultados_df = pd.DataFrame({'Creencia': [belief], 'Plausibilidad': [plausibility], 'Probabilidad_Pignistica': [pignistic]})
    resultados_df.to_csv('resultados_dempster_shafer.csv', index=False)
    st.write("Resultados de Dempster-Shafer guardados en 'resultados_dempster_shafer.csv'")

def funcionalidad_2():
    st.title('Funcionalidad 2')
    
    # Definir las variables de entrada y salida difusas
    creencia = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'creencia')
    plausibilidad = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'plausibilidad')
    probabilidad_pignistica = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'probabilidad_pignistica')

    riesgo_discrepancia = ctrl.Consequent(np.arange(0, 11, 1), 'riesgo_discrepancia')
    impacto_potencial = ctrl.Consequent(np.arange(0, 11, 1), 'impacto_potencial')
    probabilidad_ocurrencia = ctrl.Consequent(np.arange(0, 11, 1), 'probabilidad_ocurrencia')

    # Definir las funciones de membresía para cada variable
    creencia.automf(3)
    plausibilidad.automf(3)
    probabilidad_pignistica.automf(3)

    riesgo_discrepancia.automf(3)
    impacto_potencial.automf(3)
    probabilidad_ocurrencia.automf(3)

    # Visualizar las funciones de membresía
    creencia.view()
    plausibilidad.view()
    probabilidad_pignistica.view()

    riesgo_discrepancia.view()
    impacto_potencial.view()
    probabilidad_ocurrencia.view()

    st.pyplot()

    # Definir las reglas difusas
    rule1 = ctrl.Rule(creencia['poor'] & plausibilidad['poor'] & probabilidad_pignistica['poor'], 
                      (riesgo_discrepancia['poor'], impacto_potencial['poor'], probabilidad_ocurrencia['poor']))
    rule2 = ctrl.Rule(creencia['average'] & plausibilidad['average'] & probabilidad_pignistica['average'], 
                      (riesgo_discrepancia['average'], impacto_potencial['average'], probabilidad_ocurrencia['average']))
    rule3 = ctrl.Rule(creencia['good'] & plausibilidad['good'] & probabilidad_pignistica['good'], 
                      (riesgo_discrepancia['good'], impacto_potencial['good'], probabilidad_ocurrencia['good']))

    # Crear el sistema de control difuso
    riesgo_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    riesgo = ctrl.ControlSystemSimulation(riesgo_ctrl)

    # Cargar los resultados del Nivel 2
    resultados_nivel2 = pd.read_csv('resultados_dempster_shafer.csv')

    # Convertir las columnas relevantes a tipo numérico
    resultados_nivel2['Creencia'] = pd.to_numeric(resultados_nivel2['Creencia'], errors='coerce')
    resultados_nivel2['Plausibilidad'] = pd.to_numeric(resultados_nivel2['Plausibilidad'], errors='coerce')
    resultados_nivel2['Probabilidad_Pignistica'] = pd.to_numeric(resultados_nivel2['Probabilidad_Pignistica'], errors='coerce')

    # Inicializar las listas para almacenar las salidas del FIS
    riesgo_discrepancias = []
    impactos_potenciales = []
    probabilidades_ocurrencias = []

    for i in range(len(resultados_nivel2)):
        riesgo.input['creencia'] = resultados_nivel2['Creencia'][i]
        riesgo.input['plausibilidad'] = resultados_nivel2['Plausibilidad'][i]
        riesgo.input['probabilidad_pignistica'] = resultados_nivel2['Probabilidad_Pignistica'][i]
        riesgo.compute()

        riesgo_discrepancias.append(riesgo.output['riesgo_discrepancia'])
        impactos_potenciales.append(riesgo.output['impacto_potencial'])
        probabilidades_ocurrencias.append(riesgo.output['probabilidad_ocurrencia'])

    resultados_nivel3_df = pd.DataFrame({
        'Riesgo_Discrepancia': riesgo_discrepancias,
        'Impacto_Potencial': impactos_potenciales,
        'Probabilidad_Ocurrencia': probabilidades_ocurrencias
    })
    resultados_nivel3_df.to_csv('resultados_nivel3.csv', index=False)
    st.write("Resultados del Nivel 3 guardados en 'resultados_nivel3.csv'")

    # Visualización
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[0].plot(resultados_nivel3_df.index, resultados_nivel3_df['Riesgo_Discrepancia'], marker='o', linestyle='-')
    ax[0].set_xlabel('Índice')
    ax[0].set_ylabel('Riesgo Discrepancia')
    ax[0].set_title('Riesgo Discrepancia')

    ax[1].plot(resultados_nivel3_df.index, resultados_nivel3_df['Impacto_Potencial'], marker='o', linestyle='-')
    ax[1].set_xlabel('Índice')
    ax[1].set_ylabel('Impacto Potencial')
    ax[1].set_title('Impacto Potencial')

    ax[2].plot(resultados_nivel3_df.index, resultados_nivel3_df['Probabilidad_Ocurrencia'], marker='o', linestyle='-')
    ax[2].set_xlabel('Índice')
    ax[2].set_ylabel('Probabilidad de Ocurrencia')
    ax[2].set_title('Probabilidad de Ocurrencia')

    plt.tight_layout()
    st.pyplot(fig)

    # Combinación de evidencias
    mass_riesgo = MassFunction({
        'baja': np.mean(resultados_nivel3_df['Riesgo_Discrepancia']),
        'media': np.mean(resultados_nivel3_df['Impacto_Potencial']),
        'alta': np.mean(resultados_nivel3_df['Probabilidad_Ocurrencia'])
    })

    mass_impacto = MassFunction({
        'baja': np.mean(resultados_nivel3_df['Impacto_Potencial']),
        'media': np.mean(resultados_nivel3_df['Riesgo_Discrepancia']),
        'alta': np.mean(resultados_nivel3_df['Probabilidad_Ocurrencia'])
    })

    mass_probabilidad = MassFunction({
        'baja': np.mean(resultados_nivel3_df['Probabilidad_Ocurrencia']),
        'media': np.mean(resultados_nivel3_df['Riesgo_Discrepancia']),
        'alta': np.mean(resultados_nivel3_df['Impacto_Potencial'])
    })

    combined_mass = mass_riesgo & mass_impacto & mass_probabilidad

    creencia_riesgo = combined_mass.bel({'baja'})
    plausibilidad_riesgo = combined_mass.pl({'media'})
    pignistic_riesgo = combined_mass.pignistic()

    creencia_impacto = combined_mass.bel({'media'})
    plausibilidad_impacto = combined_mass.pl({'baja'})
    pignistic_impacto = combined_mass.pignistic()

    creencia_probabilidad = combined_mass.bel({'alta'})
    plausibilidad_probabilidad = combined_mass.pl({'media'})
    pignistic_probabilidad = combined_mass.pignistic()

    st.write(f"Creencia Riesgo: {creencia_riesgo}")
    st.write(f"Plausibilidad Riesgo: {plausibilidad_riesgo}")
    st.write(f"Probabilidad Pignística Riesgo: {pignistic_riesgo}")

    st.write(f"Creencia Impacto: {creencia_impacto}")
    st.write(f"Plausibilidad Impacto: {plausibilidad_impacto}")
    st.write(f"Probabilidad Pignística Impacto: {pignistic_impacto}")

    st.write(f"Creencia Probabilidad: {creencia_probabilidad}")
    st.write(f"Plausibilidad Probabilidad: {plausibilidad_probabilidad}")
    st.write(f"Probabilidad Pignística Probabilidad: {pignistic_probabilidad}")

    # Guardar resultados finales
    resultados_finales_df = pd.DataFrame({
        'Creencia_Riesgo': [creencia_riesgo],
        'Plausibilidad_Riesgo': [plausibilidad_riesgo],
        'Probabilidad_Pignistica_Riesgo': [pignistic_riesgo],
        'Creencia_Impacto': [creencia_impacto],
        'Plausibilidad_Impacto': [plausibilidad_impacto],
        'Probabilidad_Pignistica_Impacto': [pignistic_impacto],
        'Creencia_Probabilidad': [creencia_probabilidad],
        'Plausibilidad_Probabilidad': [plausibilidad_probabilidad],
        'Probabilidad_Pignistica_Probabilidad': [pignistic_probabilidad]
    })
    resultados_finales_df.to_csv('resultados_finales_nivel3.csv', index=False)
    st.write("Resultados finales del Nivel 3 guardados en 'resultados_finales_nivel3.csv'")

    # Graficar resultados finales
    labels = ['Creencia', 'Plausibilidad', 'Probabilidad Pignística']
    riesgo_vals = [creencia_riesgo, plausibilidad_riesgo, pignistic_riesgo]
    impacto_vals = [creencia_impacto, plausibilidad_impacto, pignistic_impacto]
    probabilidad_vals = [creencia_probabilidad, plausibilidad_probabilidad, pignistic_probabilidad]

    x = np.arange(len(labels))  # la etiqueta de los grupos
    width = 0.2  # el ancho de las barras

    fig, ax = plt.subplots(figsize=(12, 8))

    rects1 = ax.bar(x - width, riesgo_vals, width, label='Riesgo')
    rects2 = ax.bar(x, impacto_vals, width, label='Impacto')
    rects3 = ax.bar(x + width, probabilidad_vals, width, label='Probabilidad')

    ax.set_ylabel('Valores')
    ax.set_title('Resultados del Nivel 3 por tipo')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()
    st.pyplot(fig)


def funcionalidad_3():
    st.title('Funcionalidad 3')
    
    # Definir las variables de entrada y salida difusas
    demanda_predicha = ctrl.Antecedent(np.arange(0, 1001, 1), 'demanda_predicha')
    impacto_promocion = ctrl.Antecedent(np.arange(0, 1001, 1), 'impacto_promocion')
    inventario_rfid = ctrl.Antecedent(np.arange(0, 1001, 1), 'inventario_rfid')
    valoracion_situacion = ctrl.Consequent(np.arange(0, 11, 1), 'valoracion_situacion')

    # Definir las funciones de membresía para cada variable
    demanda_predicha['baja'] = fuzz.trimf(demanda_predicha.universe, [0, 0, 500])
    demanda_predicha['media'] = fuzz.trimf(demanda_predicha.universe, [250, 500, 750])
    demanda_predicha['alta'] = fuzz.trimf(demanda_predicha.universe, [500, 1000, 1000])

    impacto_promocion['bajo'] = fuzz.trimf(impacto_promocion.universe, [0, 0, 500])
    impacto_promocion['medio'] = fuzz.trimf(impacto_promocion.universe, [250, 500, 750])
    impacto_promocion['alto'] = fuzz.trimf(impacto_promocion.universe, [500, 1000, 1000])

    inventario_rfid['bajo'] = fuzz.trimf(inventario_rfid.universe, [0, 0, 500])
    inventario_rfid['medio'] = fuzz.trimf(inventario_rfid.universe, [250, 500, 750])
    inventario_rfid['alto'] = fuzz.trimf(inventario_rfid.universe, [500, 1000, 1000])

    valoracion_situacion['baja'] = fuzz.trimf(valoracion_situacion.universe, [0, 0, 5])
    valoracion_situacion['media'] = fuzz.trimf(valoracion_situacion.universe, [2, 5, 8])
    valoracion_situacion['alta'] = fuzz.trimf(valoracion_situacion.universe, [5, 10, 10])

    # Crear las reglas difusas
    rules = []

    for demanda_level in ['baja', 'media', 'alta']:
        for impacto_level in ['bajo', 'medio', 'alto']:
            for inventario_level in ['bajo', 'medio', 'alto']:
                if demanda_level == 'alta' or impacto_level == 'alto' or inventario_level == 'bajo':
                    valoracion_output = 'alta'
                elif demanda_level == 'media' and impacto_level == 'medio' and inventario_level == 'medio':
                    valoracion_output = 'media'
                else:
                    valoracion_output = 'baja'
                rule = ctrl.Rule(demanda_predicha[demanda_level] & impacto_promocion[impacto_level] & inventario_rfid[inventario_level], valoracion_situacion[valoracion_output])
                rules.append(rule)

    # Crear el sistema de control difuso
    valoracion_ctrl = ctrl.ControlSystem(rules)
    valoracion = ctrl.ControlSystemSimulation(valoracion_ctrl)

    # Cargar las predicciones
    predicciones_demanda = pd.read_csv('predicciones_demanda_clasificador.csv')
    predicciones_impacto_promocion = pd.read_csv('predicciones_inventario_necesario.csv')
    predicciones_inventario_rfid = pd.read_csv('predicciones_inventario_rfid.csv')

    # Inicializar la lista para almacenar las valoraciones de la situación
    valoraciones = []

    # Aplicar el FIS a cada conjunto de predicciones
    for i in range(len(predicciones_demanda)):
        valoracion.input['demanda_predicha'] = predicciones_demanda['Demanda_Predicha'][i]
        valoracion.input['impacto_promocion'] = predicciones_impacto_promocion['Inventario_Necesario'][i]
        valoracion.input['inventario_rfid'] = predicciones_inventario_rfid['Inventario_RFID'][i]

        # Calcular la valoración de la situación
        valoracion.compute()

        # Agregar la valoración a una lista
        valoraciones.append(valoracion.output['valoracion_situacion'])

    # Utilizar la biblioteca pyds para combinar las valoraciones con Dempster-Shafer
    mfs = [MassFunction({'low': v/10, 'medium': 1 - v/10, 'high': 0}) for v in valoraciones]
    combined_mf = mfs[0]
    for mf in mfs[1:]:
        combined_mf = combined_mf.combine_conjunctive(mf)

    # Convertir las claves del diccionario a cadenas de texto para evitar errores de serialización
    belief = combined_mf.bel()
    belief_str = {str(key): value for key, value in belief.items()}
    
    st.write("Valor de combined_mf.bel():", belief_str)

    # Calcular los valores para la gráfica
    labels = ['low', 'medium', 'high']
    values = [belief.get(frozenset([label]), 0) for label in labels]

    # Verificar que los valores no son todos ceros
    if all(v == 0 for v in values):
        st.write("Advertencia: Todos los valores de la gráfica son ceros. Verifica que las predicciones y el cálculo de las valoraciones sean correctos.")
    else:
        # Graficar la combinación Dempster-Shafer
        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_title('Combinación Dempster-Shafer de Valoraciones')
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Grado de Creencia')
        st.pyplot(fig)

def funcionalidad_4():
    st.title('Funcionalidad 4')
    
    # Definir las variables de entrada y salida difusas
    demanda_predicha = ctrl.Antecedent(np.arange(0, 1001, 1), 'demanda_predicha')
    impacto_promocion = ctrl.Antecedent(np.arange(0, 1001, 1), 'impacto_promocion')
    inventario_rfid = ctrl.Antecedent(np.arange(0, 1001, 1), 'inventario_rfid')
    valoracion_situacion = ctrl.Consequent(np.arange(0, 11, 1), 'valoracion_situacion')

    # Definir las funciones de membresía para cada variable
    demanda_predicha['baja'] = fuzz.trimf(demanda_predicha.universe, [0, 0, 500])
    demanda_predicha['media'] = fuzz.trimf(demanda_predicha.universe, [250, 500, 750])
    demanda_predicha['alta'] = fuzz.trimf(demanda_predicha.universe, [500, 1000, 1000])

    impacto_promocion['bajo'] = fuzz.trimf(impacto_promocion.universe, [0, 0, 500])
    impacto_promocion['medio'] = fuzz.trimf(impacto_promocion.universe, [250, 500, 750])
    impacto_promocion['alto'] = fuzz.trimf(impacto_promocion.universe, [500, 1000, 1000])

    inventario_rfid['bajo'] = fuzz.trimf(inventario_rfid.universe, [0, 0, 500])
    inventario_rfid['medio'] = fuzz.trimf(inventario_rfid.universe, [250, 500, 750])
    inventario_rfid['alto'] = fuzz.trimf(inventario_rfid.universe, [500, 1000, 1000])

    valoracion_situacion['baja'] = fuzz.trimf(valoracion_situacion.universe, [0, 0, 5])
    valoracion_situacion['media'] = fuzz.trimf(valoracion_situacion.universe, [2, 5, 8])
    valoracion_situacion['alta'] = fuzz.trimf(valoracion_situacion.universe, [5, 10, 10])

    # Crear las reglas difusas
    rules = []
    for demanda_level in ['baja', 'media', 'alta']:
        for impacto_level in ['bajo', 'medio', 'alto']:
            for inventario_level in ['bajo', 'medio', 'alto']:
                if demanda_level == 'alta' or impacto_level == 'alto' or inventario_level == 'bajo':
                    valoracion_output = 'alta'
                elif demanda_level == 'media' and impacto_level == 'medio' and inventario_level == 'medio':
                    valoracion_output = 'media'
                else:
                    valoracion_output = 'baja'
                rule = ctrl.Rule(demanda_predicha[demanda_level] & impacto_promocion[impacto_level] & inventario_rfid[inventario_level], valoracion_situacion[valoracion_output])
                rules.append(rule)

    # Crear el sistema de control difuso
    control_system = ctrl.ControlSystem(rules)
    valoracion = ctrl.ControlSystemSimulation(control_system)

    # Cargar las predicciones
    predicciones_demanda = pd.read_csv('predicciones_demanda_clasificador.csv')
    predicciones_impacto_promocion = pd.read_csv('predicciones_inventario_necesario.csv')
    predicciones_inventario_rfid = pd.read_csv('predicciones_inventario_rfid.csv')

    # Inicializar la lista para almacenar las valoraciones de la situación
    valoraciones = []

    # Aplicar el FIS a cada conjunto de predicciones
    for i in range(len(predicciones_demanda)):
        valoracion.input['demanda_predicha'] = predicciones_demanda['Demanda_Predicha'][i]
        valoracion.input['impacto_promocion'] = predicciones_impacto_promocion['Inventario_Necesario'][i]
        valoracion.input['inventario_rfid'] = predicciones_inventario_rfid['Inventario_RFID'][i]

        # Calcular la valoración de la situación
        valoracion.compute()

        # Agregar la valoración a una lista
        valoraciones.append(valoracion.output['valoracion_situacion'])

    # Utilizar la biblioteca pyds para combinar las valoraciones con Dempster-Shafer
    mfs = [MassFunction({'baja': min(v/10, 1), 'media': min(1 - v/10, 1), 'alta': 0}) for v in valoraciones]
    combined_mf = mfs[0]
    for mf in mfs[1:]:
        combined_mf = combined_mf.combine_conjunctive(mf)

    # Calcular la creencia combinada para cada categoría
    belief = combined_mf.bel()
    plausibility = combined_mf.pl()
    pignistic = combined_mf.pignistic()

    # Convertir los resultados a un formato adecuado para la visualización
    belief_serializable = {str(k): v for k, v in belief.items()}
    plausibility_serializable = {str(k): v for k, v in plausibility.items()}
    pignistic_serializable = {str(k): v for k, v in pignistic.items()}

    # Mostrar los valores de creencia, plausibilidad y probabilidad pignística en Streamlit
    st.write("Creencia:", belief_serializable)
    st.write("Plausibilidad:", plausibility_serializable)
    st.write("Probabilidad Pignística:", pignistic_serializable)

    # Graficar la combinación Dempster-Shafer
    labels = ['baja', 'media', 'alta']
    values = [belief.get('baja', 0), belief.get('media', 0), belief.get('alta', 0)]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title('Combinación Dempster-Shafer de Valoraciones')
    ax.set_xlabel('Categoría')
    ax.set_ylabel('Grado de Creencia')
    st.pyplot(fig)


def pantalla_principal():
    st.title('Presentación Taller Final.')
    if st.button('N2 D-S'):
        funcionalidad_1()
    if st.button('N3 D-S'):
        funcionalidad_2()
    if st.button('Funcionalidad 3'):
        funcionalidad_3()
    if st.button('Funcionalidad 4'):
        funcionalidad_4()

def main():
    pantalla_principal()

if __name__ == '__main__':
    main()
