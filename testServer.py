from aiohttp import web
import socketio
import pandas as pd
from pymongo import MongoClient
import datetime
from scipy.signal import argrelextrema
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from dotenv import dotenv_values
import sys
import traceback
config_var = list(dotenv_values(".env").items())
if (len(config_var) > 0):
    URL = config_var[0][1]
    DB_REAL = 'mina'
else:
    URL = "mongodb://localhost:27017"
    DB_REAL = 'databasetest'

# Funcion que Lee la base de datos y genera un DF a la salida


def read_mongo(collection='lecturas2gs', pipeline=[], flg_df=False):
    """ Read from Mongo and Store into DataFrame """
    # Connect to MongoDB
    client = MongoClient(URL)
    # Make a query to the specific DB and Collection
    result = client[DB_REAL][collection].aggregate(pipeline)
    client.close()
    list_result = list(result)
    ##print('Result DB: ',list_result)
    #list_result = list_result[0]['datos']['result']
    # Expand the cursor and construct the DataFrame
    if flg_df:
        df = pd.DataFrame(list_result)
        return df
    else:
        return list_result

# Funcion que genera el Pipline/Query para info del sitio


def genQuerySitio(sitio='SA42'):
    Rquery = [
        {
            '$match': {
                'idSitio': sitio
            }
        }, {
            '$lookup': {
                'from': 'equipos',
                'localField': 'idEquipo',
                'foreignField': 'idEquipo',
                'as': 'dataEquipo'
            }
        }, {
            '$unwind': '$dataEquipo'
        }, {
            '$project': {
                "_id": 0,
                'nombre': 1,
                'ch1': 1,
                'ch2': 1,
                'ch3': 1,
                'ch4': 1,
                'email1': 1,
                'tipo': 1,
                'intervalo': '$dataEquipo.intervalo',
                "data_alerta": 1
            }
        }
    ]
    return Rquery

# Función que Obtiene la información del sitio y hace la obtención de info del sitio.


def getDataSitio(sitio, URL_Mongo):
    pipline = genQuerySitio(sitio)
    d_sitio = read_mongo('sitios', pipline, True)
    return d_sitio

# Función que Obtiene la información de alerta del sitio


def get_data_ventana(sitio, URL_Mongo):
    pipline = [{
        '$match': {
            'idSitio': sitio
        }
    }]
    d_sitio = read_mongo('data_alerta_cota', pipline, False)
    if (len(d_sitio) == 0):
        ##print(f'Inserte nuevo Documento AlertaVentana del Sitio: {sitio}')
        client = MongoClient(URL_Mongo)
        # Make a query to the specific DB and Collection
        client[DB_REAL]['data_alerta_cota'].insert_one({
            "idSitio": sitio,
            "data_tendencia": [None, None, None, None],
            "fechalectura": [None, None, None, None]})
        client.close()
    d_sitio = read_mongo('data_alerta_cota', pipline, False)
    return d_sitio

# Genera el Pipline/Query para obtener las lecturas necesarias


def genQueryLec(sitio='SA42',
                fecha=datetime.datetime(
                    2021, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
                horas=1,str_fch = 'fechalectura'):
    intr_fch = {
        '$gte': fecha - datetime.timedelta(hours=horas),
        '$lte': fecha
    }
    match_dict = {'idSitio': sitio}
    match_dict[str_fch] = intr_fch
    Rquery = [
        {
            '$match': match_dict
        }
    ]
    return Rquery

# Obtiene el DF General de las lecturas


def getDFGlobal(sitio, collection, fechalec, intrv,str_f):
    horas = int(intrv)*3
    pipline = genQueryLec(sitio, fechalec, horas,str_f)
    df_concat = pd.DataFrame(read_mongo(collection, pipline, True))
    #print(f'DF = {df_concat.head()}')
    return df_concat

# Obtiene la información de RL por cada dato ingresado de MA


def get_lr(indx, dataMA, intvl, mins_lec, d_crudo, int_aux, deci_conf, dec_np, int_np):
    # print(f'Entre a get LR \n dato_crudo = {d_crudo}')
    lr = LinearRegression()
    serie_X = pd.Series(list(range(indx, len(dataMA))))
    serie_Y = dataMA.iloc[indx:]
    lr.fit(serie_X.to_frame(), serie_Y)
    #print(f'Fit de LR')
    serie_X2 = pd.Series(
        list(range(indx, len(dataMA)+int(intvl/int(60/mins_lec)))))
    coefficient = lr.coef_[0]
    angulo = math.degrees(math.atan(coefficient))
    r2 = lr.score(serie_X.to_frame(), serie_Y)
    dato_limite = lr.predict(pd.Series(len(dataMA)+intvl).to_frame())[0]
    d_crudo_lr = (dato_limite/100) + int_aux
    flg_n_p = False
    dato_crud_ult = 0
    if (type(d_crudo) != np.float64):
        deci_calc = abs(d_crudo_lr - d_crudo.iloc[-1])
        '''n = len(d_crudo)
        for i in d_crudo:
            if abs(i - d_crudo.iloc[-1]) > dec_np:
                flg_n_p = True
            n = n-1'''
        dato_crud_ult = d_crudo.iloc[-1]
    else:
        dato_crud_ult = d_crudo
        deci_calc = abs(d_crudo_lr - d_crudo)
    flg_p = deci_calc > deci_conf

    # print(f'''Comparando con el pico de indice: {indx}
    # m = {coefficient}
    # signo tendencia = {np.sign(coefficient)}
    # Angulo = {angulo}
    # R2: {r2}
    # Deci Calc = {deci_calc} VS Deci Config= {deci_conf}''')
    resp_dict = {
        'coefficient': coefficient,
        'signo': np.sign(coefficient),
        'angulo': angulo,
        'r2': r2,
        'deci_calc_pred': deci_calc,
        'dato_limite': d_crudo_lr,
        'dato_crud_ult': dato_crud_ult,
        'flgs': {
            'predict': flg_p,
            'near_peak': flg_n_p
        }
    }
    return resp_dict

# Obtiene Letra de Cada Cota por Canal


def get_letter_ch(cota_ch):
    switcher = {
        'cota_1': "- A ",
        'cota_2': "- B ",
        'cota_3': "- C ",
        'cota_4': "- D ",
        'cota_5': " ",
    }
    return switcher.get(cota_ch, " ")

# Obtiene el texto deaacuerdo al signo de la tendencia


def tipoAlerta(signo):
    switcher = {
        1: " ascendente ",
        0: " horizontal ",
        -1: " descendente ",
        -0: " horizontal ",
    }
    return switcher.get(signo, "Normal")

# Obtiene Mensaje de Alerta deacuerdo a los datos.


def getMensaje(fecha, cota_name, data_cota, nombre_sit, intervalo, decimales, int_np, flg_a):
    letter_ch = get_letter_ch(cota_name)
    msj = ""
    if (data_cota['flgs']['predict'] and flg_a):
        msj = msj + \
            f"Fecha:{fecha}\nSitio: {nombre_sit}{letter_ch}: Alerta de tendencia {tipoAlerta(data_cota['signo'])}, el cual si se mantiene así, llegará en {intervalo} horas al límite de cambio establecido en ese intervalo de tiempo\n"
    if (data_cota['flgs']['near_peak'] and flg_a):
        msj = msj + f"Fecha:{fecha}\nSitio: {nombre_sit}{letter_ch}: Alerta de cambio abrupto {tipoAlerta(data_cota['signo_np'])}, el sitio tuvo un cambio dentro de un intervalo de {int_np} horas anteriores mayor a {decimales} con respecto a la ultima lectura.\n"

    return msj

# Obtiene las banderas si es necesario mandar la alerta via Email.


def get_flg_predict_np(fecha_act, data_ant, f_ant, ch, sitio, data_act, intvl):
    # print(
    #    f'Generar Banderas NP y Predict\nFechaAct {fecha_act}\n Datos Ant: {data_ant}\n Datos Act: {data_act}')
    flg_a = False
    flg_predict = False
    flg_np = False
    if (data_ant != None):
        ##print(f'No es none \n')
        diff = abs(fecha_act.replace(tzinfo=datetime.timezone.utc) -
                   f_ant.replace(tzinfo=datetime.timezone.utc))
        #print(f'Diff Time: {diff}, tipo {type(diff)}')
        delta_t = diff / datetime.timedelta(hours=1)

        if (data_act['flgs']['predict']):
            if (intvl < delta_t) or (data_ant['flgs']['predict'] == False) or (data_ant['signo'] != data_act['signo']):
                flg_predict = True
        if (data_act['flgs']['near_peak']):
            if (intvl < delta_t) or (data_ant['flgs']['near_peak'] == False) or (data_ant.get('signo_np',False) != data_act['signo']):
                flg_np = True
        if flg_np or flg_predict:
            #print('Genero Alerta')
            update_data_alerta(ch, data_act, URL, sitio,
                               fecha_act, flg_np, flg_predict, False)
            flg_a = True
    else:
        # print(f'None')
        if (data_act['flgs']['predict']):
            flg_predict = True
        if (data_act['flgs']['near_peak']):
            flg_np = True
        if flg_np or flg_predict:
            #print('Genero Alerta')
            update_data_alerta(ch, data_act, URL, sitio,
                               fecha_act, flg_np, flg_predict, True)
            flg_a = True

    return flg_a, flg_predict, flg_np

# Actualiza los datos de alerta para tener una referencia.


def update_data_alerta(ch, data_act, urlDB, sitio, fecha, f_np, f_p, f_isnone):
    ##print(""" Search and Update Data in Mongo""")
    # Connect to MongoDB
    client = MongoClient(urlDB)
    # Make a query to the specific DB and Collection
    d_alerta = list(client[DB_REAL]['data_alerta_cota'].find({
        "idSitio": sitio}))[0]
    if (not(f_isnone)):

        if d_alerta['data_tendencia'][ch]['flgs']['predict']:
            data_act['flgs']['predict'] = True
        else:
            data_act['flgs']['predict'] = f_p

        if d_alerta['data_tendencia'][ch]['flgs']['near_peak']:
            data_act['flgs']['near_peak'] = True
        else:
            data_act['flgs']['near_peak'] = f_np
    else:
        data_act['flgs']['predict'] = bool(data_act['flgs']['predict'])
        data_act['flgs']['near_peak'] = bool(data_act['flgs']['near_peak'])
    ##print(f'Alerta {d_alerta}')
    d_alerta['data_tendencia'][ch] = data_act
    d_alerta['fechalectura'][ch] = fecha
    del d_alerta['_id']
    ##print(f'Update {d_alerta}')

    client[DB_REAL]['data_alerta_cota'].update_one(
        {"idSitio": sitio}, {"$set": d_alerta})

    client.close()

# Obtiene la  bandera Cambio Abrupto con signo


def getNPFlg(d_crudo, dec_np):
    flg_ini = True
    flg_np = False
    if (type(d_crudo) != np.float64):
        for i in d_crudo:
            dec_calc = d_crudo.iloc[-1] - i
            if flg_ini:
                aux_dec_calc = dec_calc
                flg_ini = False

            if abs(dec_calc) > dec_np:
                flg_np = True
            if abs(dec_calc) > abs(aux_dec_calc):
                aux_dec_calc = dec_calc
        signo_np = np.sign(aux_dec_calc)
    else:
        flg_np = False
        signo_np = False

    return flg_np, signo_np

# Obtiene el total de alertas por cada Canal


def getAlertaV2(df, num_cota, intervalo, mins_intv, decim_canal, dec_np, int_np, name_sitio, fecha_act, data_ant, f_ant, ch, sitio, strfch):
    if intervalo <= 8:
        int_ma = 4
    elif intervalo <= 24:
        int_ma = 8
    else:
        int_ma = 24
    data_ncota = np.array(df[num_cota])
    for num, cota in enumerate(data_ncota, start=0):
        if pd.isna(cota) or (cota == None):
            if num == 0:
                for uni_dat in data_ncota[num:]:
                    if not(pd.isna(uni_dat)) and uni_dat != None:
                        data_ncota[num] = uni_dat
                        break
            else:
                data_ncota[num] = data_ncota[num-1]
    df_maux = pd.Series(data_ncota)
    int_aux = math.modf(df[num_cota][0])[1]+1
    data_aux = (df_maux - int_aux)*100
    data_ma = data_aux.rolling(int_ma).mean()
    maximums = argrelextrema(np.array(data_ma), np.greater, order=5)
    minimums = argrelextrema(np.array(data_ma), np.less, order=5)
    indice_peak = np.sort(np.append(minimums[0], maximums[0]))
    # print(
    #    f'Los indices de cambio de tendencia son: {indice_peak}, tamaño de muestra: {len(data_ma)}')
    data = []
    flg_not_r2_pass = False
    flg_peak_val = True
    val_rlaux = None
    if len(indice_peak) > 0:
        #print('Hay cambios de tendencia')
        for i in range(1, len(indice_peak)+1):
            if int(3) <= abs(len(data_ma)-indice_peak[-i]):
                val_rl = get_lr(
                    indice_peak[-i], data_ma, intervalo, mins_intv, df[num_cota].iloc[-int(int_np*4):], int_aux, decim_canal, dec_np, int_np)
                #print( 'R2' + str(val_rl['r2']))
                if val_rl['r2'] > 0.85:
                    val_rl['fechalec'] = df[strfch].iloc[indice_peak[-i]]
                    #val_rl['fechalec'] = fecha_act
                    #print(f'Val_RL Append')
                    data.append(val_rl)
                    flg_peak_val = False
                else:
                    flg_not_r2_pass = True
                if val_rlaux == None:
                    val_rlaux = val_rl
                else:
                    if val_rlaux['r2'] < val_rl['r2']:
                        val_rlaux = val_rl
            else:
                flg_not_r2_pass = True

    if (len(indice_peak) <= 0 or flg_not_r2_pass) and flg_peak_val:
        #print('No Hay Cambios')
        val_rl = get_lr(intervalo, data_ma, intervalo, mins_intv,
                        df[num_cota].iloc[-1], int_aux, decim_canal, dec_np, int_np)
        val_rl['fechalec'] = df[strfch].iloc[-1]
        ##print(f'Val_RL= {val_rl}')
        if val_rlaux != None:
            if val_rl['r2'] < val_rlaux['r2']:
                data.append(val_rlaux)
            else:
                data.append(val_rl)
        else:
            data.append(val_rl)
    #print(f'Data_append {data} r2_flg {flg_not_r2_pass} flg_peak_val {flg_peak_val} len<0 {len(indice_peak) <= 0}')
    i = 0
    aux = data[0]['r2']
    idx_ch = 0
    flg_a = False
    for idx_peak, data_ch in enumerate(data, start=0):
        if aux <= data_ch['r2']:
            aux = data_ch['r2']
            idx_ch = idx_peak
    flg_np, signo_np = getNPFlg(
        df[num_cota].iloc[-int(int_np*(60/mins_intv)):], dec_np)
    data[idx_ch]['signo_np'] = signo_np
    data[idx_ch]['flgs']['near_peak'] = flg_np
    print(f'Data = {data[idx_ch]}')
    flg_a, _, _ = get_flg_predict_np(
        fecha_act, data_ant, f_ant, ch, sitio, data[idx_ch], intervalo)
    fecha_str = fecha_act.strftime("%d/%m/%Y, %H:%M:%S")
    salida = {
        'flg_a': flg_a,
        'data_r': data[idx_ch],
        'data_t': data,
        'msj': getMensaje(fecha_str, num_cota, data[idx_ch], name_sitio, intervalo, dec_np, int_np, flg_a)
    }
    ##print(f'Salida GetAlertaV2 = {salida}')
    return salida

# Coloca la Alerta en la DB


def put_alert_sitio(collection, urlDB, msj, idSitio, fechalec, email_adr, name_sit):
    """ Put Data in Mongo"""
    # Connect to MongoDB
    client = MongoClient(urlDB)
    # Make a query to the specific DB and Collection
    client[DB_REAL][collection].insert_one({
        "idSitio": idSitio,
        "nomSitio": name_sit,
        "pending": True,  # False: Salen Correos True: No salen Correos
        "tipo": 'AlertaCotaPred',
        "mensaje": msj,
        "email": False,
        "e1": email_adr,
        "fechalectura": fechalec,
        "fechainsert": datetime.datetime.now()})
    client.close()


# creates a new Async Socket IO Server
sio = socketio.AsyncServer()
# Creates a new Aiohttp Web Application
app = web.Application()
# Binds our Socket.IO server to our Web App
# instance
sio.attach(app)

# If we wanted to create a new websocket endpoint,
# use this decorator, passing in the name of the
# event we wish to listen out for


@sio.on('rx_alerta')
# Funcion de llamada de RX Para generar la alerta de Cota
async def gen_alerta(sid, data):
    # When we receive a new event of type
    # 'message' through a socket.io connection
    # we #print the socket ID and the message
    #print("Socket ID: ", sid)
    print(f'\n Data llegando del WeSocket: {data}')
    DATE_TIME_STRING_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
    sitio = data['idSitio']
    dSitio = getDataSitio(sitio, URL)
    data_ventana = get_data_ventana(sitio, URL)[0]
    #print(f'data_ventana = {data_ventana} \n dSitio = {dSitio}')
    fechalec = datetime.datetime.strptime(
        data['fechalec'], DATE_TIME_STRING_FORMAT)
    data_a = dSitio.iloc[0]['data_alerta']
    tipoSit = dSitio.iloc[0]["tipo"]
    min_intv = dSitio.iloc[0]["intervalo"]
    data_alerta_ch = []
    msj_tot = ""
    flg_msj = False
    
    try:
        if tipoSit == 4:
            for nmc, msd in enumerate(dSitio.drop(columns=['data_alerta', 'nombre', "email1", "intervalo", "tipo"]).columns, start=0):
                if dSitio.iloc[0][msd]:
                    if data_a['io'][nmc]:
                        # print(
                        # f'\n\n Tipo 4 - {msd} Activo, Requiere Calcul de #Alertas, indice {nmc}')
                        interval = data_a['intervalo'][nmc]
                        num_cota = 'cota_'+str(nmc+1)
                        decim_ch = data_a['decimales'][nmc]
                        coleccion = "lecturas2gs"
                        fecha_alerta_ant = data_ventana['fechalectura'][nmc]
                        data_alerta_ant = data_ventana['data_tendencia'][nmc]
                        dec_np = data_a['dec_np'][nmc]
                        int_np = data_a['int_np'][nmc]
                        globalDF = getDFGlobal(
                            sitio, coleccion, fechalec, interval, 'fechalectura')
                        data_ch = getAlertaV2(
                            globalDF, num_cota, interval, min_intv, decim_ch, dec_np, int_np, dSitio.iloc[0]['nombre'], fechalec, data_alerta_ant, fecha_alerta_ant, nmc, sitio, 'fechalectura')
                        data_alerta_ch.append(data_ch)
                        if (data_ch['flg_a']):
                            msj_tot = msj_tot + data_ch['msj']
                            flg_msj = True
        elif tipoSit == 9:
            if dSitio.iloc[0]["ch1"]:
                if data_a['io'][0]:
                    #print(f'Tipo 9 - CH1 Activo, Requiere Calcul de Alertas')
                    interval = data_a['intervalo'][0]
                    num_cota = 'd3'
                    decim_ch = data_a['decimales'][0]
                    coleccion = 'lecturasgs'
                    fecha_alerta_ant = data_ventana['fechalectura'][0]
                    data_alerta_ant = data_ventana['data_tendencia'][0]
                    dec_np = data_a['dec_np'][0]
                    int_np = data_a['int_np'][0]
                    globalDF = getDFGlobal(
                        sitio, coleccion, fechalec, interval, 'tf')
                    data_ch = getAlertaV2(
                        globalDF, num_cota, interval, min_intv, decim_ch, dec_np, int_np, dSitio.iloc[0]['nombre'], fechalec, data_alerta_ant, fecha_alerta_ant, 0, sitio, 'tf')
                    data_alerta_ch.append(data_ch)
                    if (data_ch['flg_a']):
                        msj_tot = msj_tot + data_ch['msj']
                        flg_msj = True
        else:
            if dSitio.iloc[0]["ch1"]:
                if data_a['io'][0]:
                    #print(f'Tipo 3 - CH1 Activo, Requiere Calcul de Alertas')
                    interval = data_a['intervalo'][0]
                    num_cota = 'cota'
                    decim_ch = data_a['decimales'][0]
                    coleccion = "lecturas2"
                    fecha_alerta_ant = data_ventana['fechalectura'][0]
                    data_alerta_ant = data_ventana['data_tendencia'][0]
                    dec_np = data_a['dec_np'][0]
                    int_np = data_a['int_np'][0]
                    globalDF = getDFGlobal(
                        sitio, coleccion, fechalec, interval, 'fechalectura')
                    data_ch = getAlertaV2(
                        globalDF, num_cota, interval, min_intv, decim_ch, dec_np, int_np, dSitio.iloc[0]['nombre'], fechalec, data_alerta_ant, fecha_alerta_ant, 0, sitio, 'fechalectura')
                    data_alerta_ch.append(data_ch)
                    if (data_ch['flg_a']):
                        msj_tot = msj_tot + data_ch['msj']
                        flg_msj = True
        # #print(
        #    f'Data Alerta Total: {data_alerta_ch}\nDataAlerta = {len(data_alerta_ch)}')
        if (flg_msj):
            print(f'Hubo alerta del Sitio: {sitio}\nMensaje: {msj_tot}')
            put_alert_sitio('alertas', URL, msj_tot, sitio, fechalec,
                            dSitio.iloc[0]['email1'], dSitio.iloc[0]['nombre'])
        else:
            print(f'Sin ALERTAS del Sitio: {sitio}')
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        print(f'\nTrono con el sitio {sitio}, error {e} en la linea {exc_tb.tb_lineno}')
        print(traceback.format_exc())

# We kick off our server
if __name__ == '__main__':
    web.run_app(app, port=6000)
