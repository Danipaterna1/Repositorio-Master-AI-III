import os
from dotenv import load_dotenv
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

# --- 1. Configuración Inicial ---
# Carga las variables de entorno (tu API key de OpenAI)
load_dotenv()

# --- CONSTANTES ---
# Definir el modelo como una constante para facilitar cambios futuros
MODEL_NAME = "gpt-4o"

# Crea y parchea el cliente de OpenAI con instructor.
# Esto le da al cliente la capacidad de devolver modelos de Pydantic.
client = instructor.patch(OpenAI())

# --- 2. Definición de las Estructuras de Datos con Pydantic ---
class InformacionEmail(BaseModel):
    """Modelo para extraer la información clave de un email de solicitud."""
    pedido_id: str = Field(..., description="Número de identificación único del pedido, ej: #D347-STELLA")
    cliente: str = Field(..., description="Nombre completo del remitente del email.")
    contacto: str = Field(..., description="Email o cualquier otra forma de contacto proporcionada.")
    motivo: str = Field(..., description="Resumen breve del motivo principal de la solicitud (ej: 'Daños en transporte').")
    detalles: str = Field(..., description="Extracto de los detalles relevantes del email que ayuden a tomar una decisión.")

class DecisionDevolucion(BaseModel):
    """Modelo para la toma de decisión sobre una solicitud de devolución."""
    aceptar: bool = Field(..., description="True si la devolución se acepta, False si se rechaza.")
    razon: str = Field(..., description="Explicación clara y concisa de por qué se tomó la decisión. Esta razón se usará en el email de respuesta al cliente.")

class EmailRespuesta(BaseModel):
    """Modelo para generar el cuerpo final del email de respuesta."""
    cuerpo_email: str = Field(..., description="El texto completo y profesional del email de respuesta, listo para ser enviado.")

# --- 3. El Workflow Automatizado ---

def procesar_solicitud_devolucion(email: str):
    """
    Flujo de trabajo completo para procesar un email de solicitud de devolución.
    Paso 1: Extrae la información estructurada del email.
    Paso 2: Toma una decisión basada en la información extraída.
    Paso 3: Redacta un email de respuesta profesional.
    """
    print("--- PASO 1: Extrayendo información del email ---")
    try:
        info_extraida = client.chat.completions.create(
            model=MODEL_NAME, # Usando la constante
            response_model=InformacionEmail,
            messages=[
                {"role": "user", "content": f"Extrae la información clave del siguiente email:\n\n{email}"}
            ]
        )
        print("Información extraída con éxito:")
        print(info_extraida.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error en la extracción de información: {e}")
        return

    print("\n--- PASO 2: Tomando una decisión ---")
    try:
        decision = client.chat.completions.create(
            model=MODEL_NAME, # Usando la constante
            response_model=DecisionDevolucion,
            messages=[
                {"role": "system", "content": """
                    Eres un agente de soporte que debe decidir si aceptar o rechazar una devolución.
                    REGLAS PARA ACEPTAR:
                    - Defecto de fabricación confirmado.
                    - Error en el suministro (modelo, cantidad o especificación incorrecta).
                    - Producto incompleto de fábrica.

                    REGLAS PARA RECHAZAR:
                    - Daños ocasionados durante el transporte (si no fue asegurado por nosotros).
                    - Manipulación indebida por parte del cliente.
                    - Solicitud fuera del plazo de devoluciones.

                    Analiza el motivo y los detalles y toma una decisión.
                """},
                {"role": "user", "content": f"Motivo de la solicitud: {info_extraida.motivo}\nDetalles: {info_extraida.detalles}"}
            ]
        )
        print("Decisión tomada:")
        print(decision.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error en la toma de decisión: {e}")
        return

    print("\n--- PASO 3: Redactando la respuesta ---")
    try:
        respuesta = client.chat.completions.create(
            model=MODEL_NAME, # Usando la constante
            response_model=EmailRespuesta,
            messages=[
                {"role": "system", "content": "Escribe un email de respuesta formal y cordial. Firma como 'Atención al Cliente de CII'."},
                {"role": "user", "content": f"""
                    Redacta un email para el cliente {info_extraida.cliente} sobre su pedido {info_extraida.pedido_id}.
                    La decisión sobre su solicitud es: {'ACEPTADA' if decision.aceptar else 'RECHAZADA'}.
                    La justificación es la siguiente: {decision.razon}.
                """}
            ]
        )
        print("Respuesta generada:")
        return respuesta.cuerpo_email
    except Exception as e:
        print(f"Error en la redacción de la respuesta: {e}")
        return

# --- Email de ejemplo ---

email_de_ejemplo = (
    "Asunto: Solicitud de reemplazo por daños en transporte – Pedido #D347-STELLA\n"
    "Estimado equipo de Componentes Intergalácticos Industriales S.A.,\n"
    "Me pongo en contacto con ustedes como cliente reciente para comunicar una "
    "incidencia relacionada con el pedido #D347-STELLA, correspondiente a un lote de "
    "condensadores de fluzo modelo FX-88, destinados a un proyecto estratégico de gran "
    "envergadura: la construcción de la Estrella de la Muerte.\n"
    "Lamentablemente, al recibir el envío, observamos que varios de los condensadores "
    "presentaban daños visibles y no funcionales. Tras revisar el estado del embalaje y "
    "consultar con el piloto de carga, todo indica que la mercancía sufrió una caída "
    "durante el transporte interestelar.\n"
    "Dado que estos componentes son críticos para la activación del núcleo central del "
    "sistema de rayos destructores, les solicitamos con carácter urgente el reemplazo "
    "inmediato de las unidades defectuosas, así como una revisión de los protocolos de "
    "embalaje y transporte para evitar que algo así vuelva a ocurrir.\n"
    "Adjunto imágenes del estado de los condensadores y el albarán de entrega sellado "
    "por nuestro droide de recepción.\n"
    "Agradezco de antemano su pronta atención a este asunto. Quedamos a la espera de "
    "su respuesta para coordinar el reemplazo.\n"
    "Atentamente,\n"
    "Darth Márquez\n"
    "Departamento de Ingeniería Imperial\n"
    "Sector de Proyectos Especiales\n"
    "Contacto: dmarquez@imperiumgalactic.net\n"
    "Holofono: +34 9X9 123 456"
)

# --- Ejecución del workflow ---
# CORRECCIÓN: Se usa __name__ == "__main__" con dobles guiones bajos.
if __name__ == "__main__":
    respuesta_final = procesar_solicitud_devolucion(email_de_ejemplo)
    if respuesta_final:
        print("\n" + "="*50)
        print("EMAIL DE RESPUESTA FINAL")
        print("="*50)
        print(respuesta_final)
