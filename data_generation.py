# build_ar_police_normal.py
# Genera N datos sintéticos policiales argentinos usando Responses API (modo normal).
# Guarda UN objeto JSON por línea con la clave "messages" (system, user, assistant).
#
# Env:
#   OPENAI_API_KEY=sk-...
#   MODEL=gpt-5-mini
#   N=1000
#   CONCURRENCY=8
#   MAX_RETRIES=5
#   ENABLE_WEB_SEARCH=true|false

import os, json, time, random, datetime, threading, re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- config ----------------
OUTDIR = Path("ar_police_normal"); OUTDIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH  = OUTDIR / "results.jsonl"
ERRORS_PATH   = OUTDIR / "errors.jsonl"

# few-shots: primero ../data/train.jsonl (como pediste), si no, data/train.jsonl
TRAIN_CANDIDATES = [Path("data/train.jsonl"), Path("data/train.jsonl")]

MODEL         = os.getenv("MODEL", "gpt-5-mini")
N             = int(os.getenv("N", "5000"))
SEED          = 20250904
FEW_SHOT_N    = int(os.getenv("FEW_SHOT_N", "6"))
FEW_SHOT_MAX  = int(os.getenv("FEW_SHOT_MAXCHARS", "700"))
CONCURRENCY   = int(os.getenv("CONCURRENCY", "32"))
MAX_RETRIES   = int(os.getenv("MAX_RETRIES", "5"))
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "false").lower() == "true"

random.seed(SEED)

# ---------------- utilidades de dominio ----------------
barrios = ["Palermo","Balvanera","Caballito","Flores","Boedo","Núñez","Almagro",
           "Recoleta","Chacarita","Constitución","Parque Patricios","Retiro",
           "San Telmo","Villa Urquiza"]
calles  = ["Av. Rivadavia","Av. Corrientes","Av. Santa Fe","Av. La Plata","Av. Cabildo",
           "Av. Del Libertador","Güemes","Scalabrini Ortiz","Billinghurst","Medrano",
           "Gaona","Nazca","Triunvirato","Díaz Vélez","Warnes","Jujuy","Belgrano"]
letras  = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
nums    = "0123456789"
lunfardo = ["guita","punga","chorro","afanar","tira","cana","escruche","fierro",
            "motochorro","rati","salidera","tumbera","linyera","buchón","laburo",
            "garufa","gayola","mina","mechera","salidera"]
ordinales = ["cero","primero","segundo","tercero","cuarto","quinto","sexto","septimo","octavo","noveno"]

def pick(x): return random.choice(x)
def hora_rand():
    h = random.randint(0,23); m = random.choice([0,5,10,15,20,25,30,35,40,45,50,55])
    return f"{h:02d}:{m:02d}"
def direccion_rand(): return f"{pick(calles)} {random.randint(100, 9900)}"
def fecha_iso_hoy(): return datetime.date.today().isoformat()
def patente_mercosur(): return "".join(random.choices(letras,k=2))+ "".join(random.choices(nums,k=3))+ "".join(random.choices(letras,k=2))
def patente_vieja():    return "".join(random.choices(letras,k=3))+ "".join(random.choices(nums,k=3))

fonetico_libre = {
    "A": ["árbol","ana","auto","antonio"], "B": ["bastón","boca","berta"], "C": ["casa","carlos","codo"],
    "D": ["dado","dora"], "E": ["estrella","eduardo"], "F": ["foco","francisco"], "G": ["gato","gustavo"],
    "H": ["hacha","hector"], "I": ["isla","irma"], "J": ["jugo","jorge"], "K": ["kilo"],
    "L": ["luna","luis"], "M": ["mano","miguel"], "N": ["nube","norma"], "O": ["oso","oscar"],
    "P": ["pato","pedro"], "Q": ["queso","quintana"], "R": ["ratón","roberto"], "S": ["silla","santiago"],
    "T": ["taza","tomas"], "U": ["uña","uruguay"], "V": ["vaca","venezuela"], "W": ["washington"],
    "X": ["xilofón"], "Y": ["yolanda"], "Z": ["zapato"]
}

def frase_fonetica_para_patente(p):
    out=[]
    for ch in p:
        if ch.isalpha(): out.append(pick(fonetico_libre[ch.upper()]))
        else: out.append(ordinales[int(ch)])
    return " ".join(out)

# ---------------- few-shots del train.jsonl ----------------
def load_few_shots(candidates, n: int, maxchars: int):
    for path in candidates:
        if path.exists():
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                import random as _r
                sample = _r.sample(lines, min(n, len(lines)))
                clipped = [s[:maxchars] + ("…" if len(s) > maxchars else "") for s in sample]
                block = "Ejemplos del dataset del usuario para respetar formato y tono:\n" + "\n".join(clipped)
                print(f"[FEW] Usando {path} ({len(sample)} ejemplos)")
                return block
    print("[FEW][WARN] No se encontró ../data/train.jsonl ni data/train.jsonl — seguimos sin few-shots.")
    return None

FEW_SHOT_BLOCK = load_few_shots(TRAIN_CANDIDATES, FEW_SHOT_N, FEW_SHOT_MAX)

SYSTEM_BASE = (
    "Sos un asistente policial argentino. Generá UN ejemplo sintético, breve y local. "
    "Seguí el estilo de los ejemplos del usuario si están presentes. No uses datos reales. "
    "La salida debe ser UN objeto JSON con la clave 'messages' y roles 'system','user','assistant'. "
    "NO incluyas texto extra, ni explicaciones, ni markdown, ni code fences."
)

def build_body(category, user_text):
    cat_note = {
        "fonetico": "Deletreo fonético local. Indicá 'posible' si aplica.",
        "jerga-def": "Definí el término de jerga argentina en 1-2 líneas como 'posible significado' argentino.",
        "jerga-conv": "Diálogo breve con al menos 3 jergas; cerrar con 'móvil en camino'.",
        "radio": "Radiomensaje conciso. Sin tácticas sensibles.",
        "extract_json": "Devolvé JSON válido. null si falta dato.",
        "patente-norm": "Devolvé solo la patente normalizada. Si no es válida, decilo.",
        "rag_query": "Devolvé JSON con query y filtros. Fechas relativas tipo hoy-5y.",
        "time_iso": "Devolvé ISO 8601 con -03:00.",
        "acta_contra": "JSON con infracción, lugar, hora, rubro.",
        "camaras_publicas": "JSON con recurso camaras_publicas, buffer y ventana horaria.",
        "conversacion": "Operador 911 y denunciante. Cerrar con 'móvil en camino'.",
        "direccion": "JSON con calle, altura, piso, depto, barrio, ciudad CABA.",
        "tipificacion": "Una línea, preliminar y no vinculante.",
        "fon2pat": "Frase fonética → patente AAA000 o AA000AA. Salida solo la patente normalizada.",
        "pat2fon": "Patente → frase fonética local, letras con nombres y dígitos en ordinal."
    }[category]

    sys_msgs = [{"role":"system","content": SYSTEM_BASE + " " + cat_note}]
    if FEW_SHOT_BLOCK:
        sys_msgs.append({"role":"system","content": FEW_SHOT_BLOCK})

    user_full = (
        user_text +
        " Formato OBLIGATORIO de salida (unico objeto JSON, sin comentarios): "
        "{\"messages\":[{\"role\":\"system\",\"content\":\"...\"},{\"role\":\"user\",\"content\":\"...\"},{\"role\":\"assistant\",\"content\":\"...\"}]}"
    )

    messages = sys_msgs + [{"role":"user","content": user_full}]

    body = {
        "model": MODEL,
        "input": messages,
        "max_output_tokens": 600,             # más margen para texto visible
        "reasoning": {"effort": "low"},       # reduce reasoning tokens invisibles
        "metadata": {
            "dataset": "ar-police-synth",
            "category": str(category),
            "few_shots": "true" if FEW_SHOT_BLOCK else "false"
        }
    }
    if ENABLE_WEB_SEARCH:
        body["tools"] = [{"type": "web_search"}]
        body["tool_choice"] = "auto"
    return body

def tpl_fon2pat():
    pat = random.choice([patente_vieja(), patente_mercosur()])
    frase = frase_fonetica_para_patente(pat)
    return {
        "category":"fon2pat",
        "user": f"Convertí el siguiente código fonético a una patente argentina válida: '{frase}'. "
                f"Cada palabra aporta su inicial y los números están en ordinal. Devolvé solo la patente normalizada."
    }

def tpl_pat2fon():
    pat = random.choice([patente_vieja(), patente_mercosur()])
    return {
        "category":"pat2fon",
        "user": f"Convertí la patente '{pat}' a una frase en código fonético policial argentino con ordinales para números."
    }

TEMPLATES = [
    lambda: {"category":"fonetico","user": f"Deletrea {random.choice(['RO','AA','FGH','ZULU',patente_vieja(),patente_mercosur(),'CQAP'])} en código fonético argentino como una posibilidad."},
    lambda: {"category":"jerga-def","user": f"Definí el término de jerga argentina '{pick(lunfardo)}' en contexto policial."},
    lambda: {"category":"jerga-conv","user": f"Mini conversación entre vecino y móvil por {pick(['ruidos','arrebato','daño vehicular'])} en {pick(barrios)}. "
                                            f"Usá al menos 3 de: {', '.join(random.sample(lunfardo, 5))}. Cerrá con móvil en camino."},
    lambda: {"category":"radio","user": f"Minuta de radio por {pick(['robo en proceso','conflicto vecinal','daño a propiedad','persona descompensada','ruidos molestos'])} en {direccion_rand()}, "
                                       f"sospechoso con {pick(['gorra negra','campera roja','moto azul','bicicleta','sin armas a la vista'])}."},
    lambda: {"category":"extract_json","user": f"Extraé a JSON tipo_delito, objeto, lugar, hora, autores de: "
                                              f"\"Me robaron el {pick(['celular','bolso','notebook'])} en {pick(calles)} cerca de {pick(calles)}, {hora_rand()}, {pick(['dos sujetos','uno a pie','en moto'])}\""},
    lambda: {"category":"patente-norm","user": f"Estandarizá la patente argentina '{random.choice([patente_mercosur(), patente_vieja(), 'ac 123 zz','AA12BBB'])}' a formato AAA000 o AA000AA."},
    lambda: {"category":"rag_query","user": f"Convertí a consulta RAG: \"Buscá antecedentes del DNI {''.join(random.choices(nums,k=8))} últimos 5 años en CABA y {pick(['San Martín','La Matanza','Lanús'])}\"."},
    lambda: {"category":"time_iso","user": f"Convertí a ISO BA: \"Ayer a las {pick(['6 en punto','22:35','ocho y media de la noche'])}\" con fecha de hoy {fecha_iso_hoy()}."},
    lambda: {"category":"acta_contra","user": f"Campos para acta por venta de alcohol fuera de horario en {direccion_rand()}, {hora_rand()}."},
    lambda: {"category":"camaras_publicas","user": f"Consulta RAG para cámaras públicas en {direccion_rand()} entre {hora_rand()} y {hora_rand()}."},
    lambda: {"category":"conversacion","user": f"Mini conversación de 3 turnos entre operador 911 y denunciante por {pick(['arrebato','daño de vehículo','ruidos nocturnos'])} en {pick(barrios)}. Termina con 'móvil en camino'."},
    lambda: {"category":"direccion","user": f"Normalizá \"{direccion_rand()}, {random.randint(1,8)} {pick(list('ABCDE'))}, {pick(barrios)}\" a JSON con calle, altura, piso, depto, barrio, ciudad CABA."},
    lambda: {"category":"tipificacion","user": "A partir de \"Me exigieron la billetera diciendo que tenían un arma, no la vi\", devolvé tipificación preliminar en 1 línea."},
    tpl_fon2pat,
    tpl_pat2fon,
]

def build_request(i: int):
    tpl = random.choice(TEMPLATES)()
    body = build_body(tpl["category"], tpl["user"])
    return i, tpl["category"], body

# ------------ helpers de extracción y parseo ------------
def extract_text(resp):
    # 1) atajo oficial
    try:
        txt = resp.output_text
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
    except Exception:
        pass
    # 2) bajar a dict y recorrer
    try:
        d = resp.to_dict_safe()
    except Exception:
        d = resp
    pieces = []
    try:
        for item in d.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") in ("output_text","text"):
                        t = c.get("text") or ""
                        if t: pieces.append(t)
    except Exception:
        pass
    return "\n".join(pieces).strip()

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def try_parse_messages(s: str):
    s = strip_code_fences(s)
    # intenta parsear directo
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "messages" in obj:
            return obj
    except Exception:
        pass
    # intento de recorte si vino texto extra antes/después
    m = re.search(r"\{[\s\S]*\"messages\"[\s\S]*\}\s*$", s)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "messages" in obj:
                return obj
        except Exception:
            pass
    return None

# ---------------- llamada con reintentos ----------------
def call_with_retry(i: int, category: str, body: dict):
    delay = 1.0
    last_err = None
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = client.responses.create(**body)
            text = extract_text(r)
            obj = try_parse_messages(text)
            if obj:
                return {"ok": True, "i": i, "category": category, "obj": obj}
            else:
                last_err = f"Respuesta sin JSON válido. Texto={text[:180].replace(chr(10),' ')}..."
                # en modelos razonadores, subir un poco el límite en reintento 1-2:
                if attempt <= 2 and isinstance(body.get("max_output_tokens"), int):
                    body["max_output_tokens"] = min(body["max_output_tokens"] + 200, 1200)
                # y bajar más el esfuerzo si sigue:
                if attempt == 2:
                    body["reasoning"] = {"effort":"low"}
                time.sleep(delay); delay = min(delay*2, 30)
                continue
        except Exception as e:
            last_err = str(e)
            if any(x in last_err.lower() for x in ["rate", "429", "timeout", "temporarily", "unavailable", "overloaded", "503", "504"]):
                time.sleep(delay); delay = min(delay * 2, 30); continue
            break
    return {"ok": False, "i": i, "category": category, "error": last_err}

# ---------------- main ----------------
def main():
    print(f"[CFG] MODEL={MODEL} N={N} CONCURRENCY={CONCURRENCY} WEB_SEARCH={ENABLE_WEB_SEARCH}")
    results_f = RESULTS_PATH.open("w", encoding="utf-8")
    errors_f  = ERRORS_PATH.open("w", encoding="utf-8")

    lock = threading.Lock()
    completed = 0

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futures = []
        for i in range(N):
            idx, cat, body = build_request(i)
            futures.append(ex.submit(call_with_retry, idx, cat, body))

        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                completed += 1
                if res["ok"]:
                    # GUARDA exactamente el objeto con "messages" (como pediste)
                    results_f.write(json.dumps(res["obj"], ensure_ascii=False) + "\n")
                else:
                    line = {"i": res["i"], "category": res["category"], "error": res["error"]}
                    errors_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                if completed % 50 == 0 or completed == N:
                    print(f"[PROG] {completed}/{N}")

    results_f.close(); errors_f.close()
    print(f"[OK] Guardado: {RESULTS_PATH}  |  Errores: {ERRORS_PATH}")

if __name__ == "__main__":
    main()
