-- Crear la base de datos
CREATE DATABASE keepcoding;

-- Usar la base de datos
\c keepcoding;

-- Crear la tabla Bootcamp
CREATE TABLE Bootcamp (
    id_bootcamp SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    precio DECIMAL(10,2) NOT NULL
);

-- Crear la tabla Edicion
CREATE TABLE Edicion (
    id_edicion SERIAL PRIMARY KEY,
    id_bootcamp INT NOT NULL,
    anio INT NOT NULL,
    fecha_inicio DATE NOT NULL,
    fecha_fin DATE NOT NULL,
    CONSTRAINT fk_bootcamp FOREIGN KEY (id_bootcamp) REFERENCES Bootcamp(id_bootcamp)
);

-- Crear la tabla Alumno
CREATE TABLE Alumno (
    id_alumno SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    id_edicion INT NOT NULL,
    CONSTRAINT fk_edicion_alumno FOREIGN KEY (id_edicion) REFERENCES Edicion(id_edicion)
);

-- Crear la tabla Profesor
CREATE TABLE Profesor (
    id_profesor SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    id_edicion INT NOT NULL,
    CONSTRAINT fk_edicion_profesor FOREIGN KEY (id_edicion) REFERENCES Edicion(id_edicion)
);

-- Crear la tabla Modulo
CREATE TABLE Modulo (
    id_modulo SERIAL PRIMARY KEY,
    nombre VARCHAR(255) NOT NULL,
    id_bootcamp INT NOT NULL,
    CONSTRAINT fk_bootcamp_modulo FOREIGN KEY (id_bootcamp) REFERENCES Bootcamp(id_bootcamp)
);
