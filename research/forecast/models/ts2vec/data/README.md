## INTRODUCCIÓN

En el presente documento se incluyen análisis detallados de diversos conjuntos de datos (datasets) seleccionados específicamente para su uso en aplicaciones de predicción y detección de anomalías en entornos industriales. Estos datasets permiten evaluar tanto modelos estadísticos tradicionales, como ARMA y ARIMA, que capturan patrones lineales y estacionales en series temporales, como modelos de aprendizaje profundo más avanzados, tales como Redes Neuronales Recurrentes (RNN) y Modelos de Series Temporales Lineales (TSLM), capaces de capturar relaciones no lineales y dependencias complejas en los datos.

**IMPORTANTE**: Los datasets contienen variables que no están documentadas. Por este motivo, se ha investigado su posible significado y empleado inteligencia artificial para analizarlas. n consecuencia, es posible que las interpretaciones no reflejen con exactitud la intención original de los datos.


## ÍNDICE

Los datasets seleccionados son los siguientes:
- [Electrical Measurement Dataset From a University Laboratory for Smart Energy Applications](#electricla-measurement-dataset-from-a-university-laboratory-for-smart-energy-applications)
- [Individual Household Electric Power Consumption](#individual-household-electric-power-consumption)
- [Steel Industry Energy Consumption](#steel-industry-energy-consumption)
- [Electricity Load Diagrams 2011-2014](#electricity-load-diagrams-2011-to-2014)
- [Smart Meters in London](#smart-meters-in-london)


## Electricla Measurement Dataset From a University Laboratory for Smart Energy Applications.

### Descripción

En este conjunto de datos se presentan mediciones de tensiones y corrientes trifásicas, potencia activa y reactiva (por fase y total), factor de potencia y frecuencia del sistema. Los datos fueron recopilados entre abril y diciembre de 2016.

- **SOURCE**: https://zenodo.org/records/17107426
- **REFERENCE**: [electrical_measurement_from_university_laboratory.csv](electrical_measurement_from_university_laboratory.csv)

| Num. Columnas | Num. Entradas |
|---------------|---------------|
| 20            | 33.376        |

A continuación, se especifican las columnas que contiene el conjunto de datos:

| Columna   | Tipo    | Descripción                                                                                                                    |
|-----------|---------|--------------------------------------------------------------------------------------------------------------------------------|
| Timestamp | object  | Marca temporal de la medición (YYYY-MM-DD HH:MM:SS.SSS).                                                                       |
| V_A       | float64 | Tensión de la fase A en voltios                                                                                                |
| V_B       | float64 | Tensión de la fase B en voltios                                                                                                |
| V_C       | float64 | Tensión de la fase C en voltios                                                                                                |
| I_A       | float64 | Corriente de la fase A en amperios                                                                                             |
| I_B       | float64 | Corriente de la fase B en amperios                                                                                             |
| I_C       | float64 | Corriente de la fase C en amperios                                                                                             |
| P_A       | float64 | Potencia activa de la fase A en kilovatios (kW). Representa la energía útil consumida por esa                                  |
| P_B       | float64 | Potencia activa de la fase B en kilovatios (kW). Representa la energía útil consumida por esa                                  |
| P_C       | float64 | Potencia activa de la fase C en kilovatios (kW). Representa la energía útil consumida por esa                                  |
| P_T       | float64 | Potencia activa total en kilovatios (kW).                                                                                      |
| Q_A       | float64 | Potencia reactiva de la fase A en kilovoltioamperios reactivos (kVAr). Representa energía que no realiza trabajo útil pero afecta el factor de potencia. | 
| Q_B       | float64 | Potencia reactiva de la fase B en kilovoltioamperios reactivos (kVAr). Representa energía que no realiza trabajo útil pero afecta el factor de potencia. | 
| Q_C       | float64 | Potencia reactiva de la fase C en kilovoltioamperios reactivos (kVAr). Representa energía que no realiza trabajo útil pero afecta el factor de potencia. | 
| Q_T       | float64 | Potencia reactiva total en kilovoltioamperios reactivos (kVAr).                                                                |
| FP_A      | float64 | Factor de potencia de la fase A (adimensional, entre 0 y 1). Indica la eficiencia del uso de energía activa frente a reactiva. |
| FP_B      | float64 | Factor de potencia de la fase B (adimensional, entre 0 y 1). Indica la eficiencia del uso de energía activa frente a reactiva. |
| FP_C      | float64 | Factor de potencia de la fase C (adimensional, entre 0 y 1). Indica la eficiencia del uso de energía activa frente a reactiva. |
| FP_T      | float64 | Factor de potencia total.                                                                                                      |
| F         | float64 | Frecuencia del sistema en hertzios (Hz).                                                                                       |



## Individual Household Electric Power Consumption.

Este conjunto contiene 2.075.259 mediciones recolectadas de una vivienda entre diciembre de 2006 y noviembre de 2010 (47 meses).

- **SOURCE**: https://www.kaggle.com/datasets/imtkaggleteam/household-power-consumption
- **REFERENCE**: [household_electric_power_consumption.csv](household_electric_power_consumption.csv)

| Num. Columnas | Num. Entradas |
|---------------|---------------|
| 8             | 2.075.259     |

A continuación, se especifican las columnas que contiene el conjunto de datos:

| Columna   | Tipo    | Descripción                                                                  |
|-----------|---------|------------------------------------------------------------------------------|
| Timestamp | object  | Fecha y hora de la medición (YYYY-MM-DD HH:MM:SS).                           |
| GAP       | float64 | Potencia activa global promedio por minuto del hogar en kilovatios (kW).     |
| GRP       | float64 | Potencia reactiva global promedio por minuto del hogar en kilovatios (kW).   |
| V         | float64 | Voltaje promedio por minuto en voltios (V).                                  |
| GI        | float64 | Intensidad de corriente global prmedio por minuto del hogar en amperios (A). |
| SM_1      | float64 | Submedición de energía Nº1 en vatios hora de energía activa. Corresponde a la cocina, que contiene principalmente un lavavajillas, un horno y un microondas (las placas de cociona no son eléctricas sino a gas). |
| SM_2      | float64 | Submedición de energía Nº2 en vatios hora de energía activa. Corresponde al cuarto de lavandería, que contiene una lavadora, una secadora, un refrigerador y una luz. |
| SM_3      | float64 | Submedición de energía Nº3 en vatios hora de energía activa. Corresponde a un calentador de agua eléctrico y un aire acondicionado. |


## Steel Industry Energy Consumption.

La información proviene de DAEWOO Steel Co. Ltd en Gwangyang, Corea del Sur. Dicho conjunto representa el consumo energético de esta empresa.

- **SOURCE**: https://www.kaggle.com/datasets/csafrit2/steel-industry-energy-consumption
- **REFERENCE**: [steel_industry_energy_consumption.csv](steel_industry_energy_consumption.csv)

| Num. Columnas | Num. Entradas |
|---------------|---------------|
| 11            | 35.040        |

A continuación, se especifican las columnas que contiene el conjunto de datos:

| Columna    | Tipo    | Descripción                                                                                    |
|------------|---------|------------------------------------------------------------------------------------------------|
| Timestamp  | object  | Fecha y hora de la medición (DD-MM-YYY HH:MM).                                                 |
| Ussage     | float64 | Consumo de energía de la industria en kilovatios (kW).                                         |
| LaggingCRP | float64 | Potencia reactiva retardada de la corriente en kilovoltioamperios reactivos por hora (kVArh).  |
| LeadingCRP | float64 | Potencia reactiva adelantada de la corriente en kilovoltioamperios reactivos por hora (kVArh). |
| LaggingCPF | float64 | Factor de potencia retardada.                                                                  |
| LeadingCPF | float64 | Factor de potencia adelantada.                                                                 |
| CO2        | float64 | Concentración de CO2 en ppm por minuto.                                                        |
| NSM        | float64 | Número de segundos desde medianoche en segundos (s) continuos.                                 |
| WS         | float64 | Estado de la semana (categórico: Fin de samana = 0, Día laboral = 1).                          |
| DW         | float64 | Día de la semana (categórico: domingo, lunes, ... sábado).                                     |
| LT         | float64 | Tipo de carga (categórico: Carga ligera, Carga media, Carga máxima).                           |



## Electricity Load Diagrams 2011 to 2014.

Representa el consumo energético de varios clientes. Los valores están expresados en kW cada 15 minutos. Algunos clientes fueron creados después de 2011, en esos casos, el consumo se consideró 0. Todas las etiquetas de tiempo corresponden a la hora portuguesa. Sin embargo, todos los días presentan 96 mediciones (24*4). Cada año, en marzo, el díca de cambio de hora (que tiene solo 23 horas), los valores entre la 1:00 y las 2:00 son cero para todos los puntos. Cada año, en ocubre, el día del cambio de hora (que tiene 25 horas), los valores entre la 1:00 y las 2:00 agregan el consumo de dos horas.

- **SOURCE**: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
- **REFERENCE**: [electricity_load_diagrams.csv](electricity_load_diagrams.csv)

| Num. Columnas | Num. Entradas |
|---------------|---------------|
| 371           | 140.256       |

A continuación, se especifican las columnas que contiene el conjunto de datos:

| Columna    | Tipo    | Descripción                                        |
|------------|---------|----------------------------------------------------|
| Timestamp  | object  | Fecha y hora de la medición (YYYY-MM-DD HH:MM:SS). |
| MT_N       | float   | Consumo del cliente N en kilovatios (kW).          |


## Smart Meters in London.

Conjunto de datos que contiene las lecturas de consumo energético de una muestra de 5.567 hogares londinenses que participaron en el proyecto Low Carbon London, entre noviembre de 2011 y febrero de 2015, asociados únicamente al consumo eléctrico.
También se añadieron datos meteorológicos para el área de Londres, recopilados mediante la API de Dark Sky.

- **SOURCE**: https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london
- **REFERENCE**: [smart_meters_in_london](smart_meters_in_london/)

El directorio contiene una serie de archivos:
- [*informations_households.csv*](smart_meters_in_london/informations_households.csv): Archivo que contiene toda la información sobre los hogares del panel (su grupo ACORN, su tarifa) y en que block.csv.qz se almacenan sus datos.
- [*halfhourly_dataset.zip*](smart_meters_in_london/halfhourly_dataset.zip): Archivo ZIP que contiene los archivos block con las mediciones del medidor inteligente de cada media hora.
- [*daily_dataset.zip*](smart_meters_in_london/daily_dataset.zip): Archivo ZIP que contiene los archivos block con la información diaria, como número de mediciones, mínimo, máximo, media, mediana, suma y desviación estándar.
- [*acorn_details.csv*](smart_meters_in_london/acorn_details.csv): Detalles sobre los grupos ACORN y el perfil de las personas en cada grupo, provenientes de esta hoja de cálculo XLSX. Las tres primeras columnas son los atributos estudiados; ACORN-X es el índice del atributo. A escala nacional, el índice es 100; si para una columna el valor es 150, significa que hay 1,5 veces más eprsonas con ese atributo en el grupo ACORN que a nivel nacional. (Se puede econtrar una explicación en el sitio web de CACI).
- [*weather_daily_darksy.csv*](smart_meters_in_london/weather_daily_darksky.csv): Contiene los datos diarios de la API de Dark Sky.
- [*weather_hourly_darksky.csv*](smart_meters_in_london/weather_hourly_darksky.csv.zip): Contiene los datos horarios de la API de Dark Sky.