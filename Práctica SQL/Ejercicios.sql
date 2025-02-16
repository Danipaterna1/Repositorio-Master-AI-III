CREATE DATABASE keepcoding_db;

-- EJERCICIO 2:
CREATE TABLE Bootcamp (
    id_bootcamp SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    precio DECIMAL(10, 2)
);

CREATE TABLE Modulo (
    id_modulo SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    id_bootcamp INT NOT NULL,
    FOREIGN KEY (id_bootcamp) REFERENCES Bootcamp(id_bootcamp) ON DELETE CASCADE
);

CREATE TABLE Edicion (
    id_edicion SERIAL PRIMARY KEY,
    id_bootcamp INT NOT NULL,
    anio INT NOT NULL,
    fecha_inicio DATE,
    fecha_fin DATE,
    FOREIGN KEY (id_bootcamp) REFERENCES Bootcamp(id_bootcamp) ON DELETE CASCADE
);

CREATE TABLE Profesor (
    id_profesor SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    id_edicion INT,
    FOREIGN KEY (id_edicion) REFERENCES Edicion(id_edicion) ON DELETE SET NULL
);

CREATE TABLE Alumno (
    id_alumno SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE,
    id_edicion INT,
    FOREIGN KEY (id_edicion) REFERENCES Edicion(id_edicion) ON DELETE SET NULL
);


-- EJERCICIO 3:
CREATE TABLE ivr_detail (
    calls_ivr_id VARCHAR(255) NOT NULL,
    calls_phone_number VARCHAR(255),
    calls_ivr_result VARCHAR(255),
    calls_vdn_label VARCHAR(255),
    calls_start_date TIMESTAMP WITH TIME ZONE,
    calls_start_date_id INT,
    calls_end_date TIMESTAMP WITH TIME ZONE,
    calls_end_date_id INT,
    calls_total_duration INT,
    calls_customer_segment VARCHAR(255),
    calls_ivr_language VARCHAR(255),
    calls_steps_module INT,
    calls_module_aggregation TEXT,
    module_sequence INT,
    module_name VARCHAR(255),
    module_duration INT,
    module_result VARCHAR(255),
    step_sequence INT,
    step_name VARCHAR(255),
    step_result VARCHAR(255),
    step_description_error VARCHAR(255),
    document_type VARCHAR(255),
    document_identification VARCHAR(255),
    customer_phone VARCHAR(255),
    billing_account_id VARCHAR(255)
);

-- EJERCICIO 4:

CREATE TABLE ivr_detail_con_agregacion AS
SELECT
    *,
    CASE
        WHEN calls_vdn_label LIKE 'ATC%' THEN 'FRONT'
        WHEN calls_vdn_label LIKE 'TECH%' THEN 'TECH'
        WHEN calls_vdn_label = 'ABSORPTION' THEN 'ABSORPTION'
        ELSE 'RESTO'
    END AS vdn_aggregation
FROM
    ivr_detail;

-- EJERCICIO 5:
SELECT DISTINCT ON (calls_ivr_id)
    calls_ivr_id,
    document_type,
    document_identification
FROM
    ivr_detail
WHERE
    document_type IS NOT NULL AND document_identification IS NOT NULL
ORDER BY
    calls_ivr_id,
    step_sequence;


-- EJERCICIO 6: 
SELECT DISTINCT ON (calls_ivr_id)
    calls_ivr_id,
    customer_phone
FROM
    ivr_detail
WHERE
    customer_phone IS NOT NULL
ORDER BY
    calls_ivr_id,
    step_sequence;

-- EJERCICIO 7:
SELECT DISTINCT ON (calls_ivr_id)
    calls_ivr_id,
    billing_account_id
FROM
    ivr_detail
WHERE
    billing_account_id IS NOT NULL
ORDER BY
    calls_ivr_id,
    step_sequence;

-- EJERCICIO 8:
SELECT
    i.calls_ivr_id,
    MAX(CASE WHEN i.module_name = 'AVERIA_MASIVA' THEN 1 ELSE 0 END) AS masiva_lg
FROM
    ivr_detail i
GROUP BY
    i.calls_ivr_id;

-- EJERCICIO 9:
SELECT
    i.calls_ivr_id,
    MAX(CASE WHEN i.step_name = 'CUSTOMERINFOBYPHONE.TX' AND i.step_result = 'OK' THEN 1 ELSE 0 END) AS info_by_phone_lg
FROM
    ivr_detail i
GROUP BY
    i.calls_ivr_id;

-- EJERCICIO 10:
SELECT
    i.calls_ivr_id,
    MAX(CASE WHEN i.step_name = 'CUSTOMERINFOBYDNI.TX' AND i.step_result = 'OK' THEN 1 ELSE 0 END) AS info_by_dni_lg
FROM
    ivr_detail i
GROUP BY
    i.calls_ivr_id;

-- EJERCICIO 11:
SELECT
    i.calls_ivr_id,
    MAX(CASE
        WHEN i.calls_phone_number = sub.calls_phone_number
             AND i.calls_ivr_id <> sub.calls_ivr_id
             AND sub.calls_start_date BETWEEN i.calls_start_date - INTERVAL '24 hours' AND i.calls_start_date
        THEN 1 ELSE 0
    END) AS repeated_phone_24H,
    MAX(CASE
        WHEN i.calls_phone_number = sub.calls_phone_number
             AND i.calls_ivr_id <> sub.calls_ivr_id
             AND sub.calls_start_date BETWEEN i.calls_start_date AND i.calls_start_date + INTERVAL '24 hours'
        THEN 1 ELSE 0
    END) AS cause_recall_phone_24H
FROM
    ivr_detail i
LEFT JOIN
    ivr_detail sub ON i.calls_phone_number = sub.calls_phone_number
                     AND i.calls_ivr_id <> sub.calls_ivr_id
GROUP BY
    i.calls_ivr_id;
