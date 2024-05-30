# Usar una imagen base de Python
FROM python:3.10.12-slim

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos de requisitos en el directorio de trabajo
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el contenido del proyecto en el directorio de trabajo
COPY . .
