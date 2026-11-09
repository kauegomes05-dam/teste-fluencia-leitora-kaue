import streamlit as st
import pandas as pd
import whisper
import tempfile
import os
import re
import csv
import html
import difflib
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------
# Fluência Leitora - versão inicial (MVP)
# Leitura completa: gravação, transcrição, comparação e relatório.
# ------------------------------------------------------------

st.set_page_config(page_title="Fluência Leitora | Anos Iniciais", page_icon="📚", layout="wide")
st.title("📚 Fluência Leitora — Anos Iniciais")
st.caption("Avaliação individual de leitura completa | 1º ao 5º ano")

DATA_DIR = Path("dados_fluencia")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "historico.csv"

@st.cache_resource
def carregar_whisper():
    # Modelo base oferece equilíbrio entre qualidade e desempenho.
    return whisper.load_model("base")

def normalizar_palavra(p):
    p = p.lower().strip()
    # Mantém letras e números, incluindo acentos e caracteres Unicode.
    return re.sub(r"[^\w]", "", p, flags=re.UNICODE)

def tokenizar(texto):
    return [p for p in texto.split() if p.strip()]

def extrair_texto_arquivo(upload):
    if upload is None:
        return ""
    nome = upload.name.lower()
    conteudo = upload.getvalue()
    if nome.endswith(".txt"):
        for encoding in ("utf-8", "latin-1"):
            try:
                return conteudo.decode(encoding)
            except UnicodeDecodeError:
                continue
        return ""
    if nome.endswith(".docx"):
        try:
            from docx import Document
            import io
            doc = Document(io.BytesIO(conteudo))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            st.error(f"Não foi possível ler o DOCX: {e}")
            return ""
    if nome.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(conteudo))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            st.error(f"Não foi possível ler o PDF. Verifique se é um PDF com texto selecionável. Detalhes: {e}")
            return ""
    st.warning("Formato não suportado. Use TXT, DOCX ou PDF.")
    return ""

def transcrever_audio(caminho):
    model = carregar_whisper()
    result = model.transcribe(
        caminho,
        language="pt",
        task="transcribe",
        temperature=0,
        condition_on_previous_text=False,
        word_timestamps=True
        # Não passar o texto de referência como prompt: isso pode induzir a transcrição.
    )
    return result.get("text", "").strip()

def comparar_textos(original, lido):
    orig_raw = tokenizar(original)
    lido_raw = tokenizar(lido)
    orig_norm = [normalizar_palavra(p) for p in orig_raw]
    lido_norm = [normalizar_palavra(p) for p in lido_raw]

    matcher = difflib.SequenceMatcher(None, orig_norm, lido_norm, autojunk=False)
    linhas = []
    corretas = substituicoes = omissoes = insercoes = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                linhas.append({"Esperado": orig_raw[i1+k], "Transcrição": lido_raw[j1+k], "Resultado": "Correto"})
                corretas += 1
        elif tag == "replace":
            n = min(i2-i1, j2-j1)
            for k in range(n):
                linhas.append({"Esperado": orig_raw[i1+k], "Transcrição": lido_raw[j1+k], "Resultado": "Substituição"})
                substituicoes += 1
            for k in range(n, i2-i1):
                linhas.append({"Esperado": orig_raw[i1+k], "Transcrição": "—", "Resultado": "Omissão"})
                omissoes += 1
            for k in range(n, j2-j1):
                linhas.append({"Esperado": "—", "Transcrição": lido_raw[j1+k], "Resultado": "Inserção"})
                insercoes += 1
        elif tag == "delete":
            for k in range(i1, i2):
                linhas.append({"Esperado": orig_raw[k], "Transcrição": "—", "Resultado": "Omissão"})
                omissoes += 1
        elif tag == "insert":
            for k in range(j1, j2):
                linhas.append({"Esperado": "—", "Transcrição": lido_raw[k], "Resultado": "Inserção"})
                insercoes += 1

    total_original = len(orig_raw)
    total_transcrito = len(lido_raw)
    taxa_precisao = (corretas / total_original * 100) if total_original else 0
    erros = substituicoes + omissoes + insercoes
    return {
        "linhas": linhas,
        "corretas": corretas,
        "substituicoes": substituicoes,
        "omissoes": omissoes,
        "insercoes": insercoes,
        "erros": erros,
        "total_original": total_original,
        "total_transcrito": total_transcrito,
        "precisao": taxa_precisao
    }

def estimar_tempo(ano, palavras, complexidade):
    # Estimativa orientativa, não norma oficial nem substituto de avaliação docente.
    # Faixas configuráveis de PPM aproximadas para estimar a duração total.
    faixas_ppm = {
        1: (20, 40),
        2: (40, 60),
        3: (60, 90),
        4: (80, 110),
        5: (100, 130)
    }
    min_ppm, max_ppm = faixas_ppm[ano]
    fator = {"Simples": 0.90, "Intermediário": 1.00, "Complexo": 1.20}[complexidade]
    # Complexidade aumenta o tempo estimado.
    tempo_min = palavras / max_ppm * 60 * fator
    tempo_max = palavras / min_ppm * 60 * fator
    return round(tempo_min), round(tempo_max)

def classificacao(ano, ppm, precisao):
    # Limites provisórios configuráveis. Não são referência oficial universal.
    metas = {1: 35, 2: 60, 3: 90, 4: 110, 5: 130}
    meta = metas[ano]
    razao = ppm / meta if meta else 0
    if precisao < 70 or razao < 0.50:
        nivel = "Abaixo do básico"
    elif precisao < 85 or razao < 0.75:
        nivel = "Básico"
    elif precisao < 95 or razao < 1.00:
        nivel = "Adequado"
    else:
        nivel = "Avançado"
    return nivel, meta

def orientacao(nivel, precisao, ppm, ano, meta, substituicoes, omissoes):
    if precisao < 85:
        return "Priorizar precisão: leitura compartilhada, reconhecimento de palavras, correspondência grafema-fonema e releitura orientada. Revisar manualmente os erros identificados pela transcrição."
    if ppm < meta * 0.75:
        return "Trabalhar continuidade e automatização com leitura repetida de textos curtos, modelagem pelo professor e acompanhamento do ritmo sem pressionar o estudante."
    if substituicoes + omissoes > 0:
        return "Manter práticas de leitura expressiva e atenção às palavras. Analisar as substituições e omissões no áudio, pois a transcrição automática pode apresentar falhas."
    return "Ampliar repertório textual, leitura expressiva, vocabulário e compreensão. Confirmar a classificação com observação docente e atividade de compreensão."

def salvar_historico(registro):
    df = pd.DataFrame([registro])
    if HISTORY_FILE.exists():
        df.to_csv(HISTORY_FILE, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

# ---------- Interface ----------
with st.sidebar:
    st.header("Dados da avaliação")
    professor = st.text_input("Professor / Turma", placeholder="Ex.: Prof. Kauê — 3º A")
    aluno = st.text_input("Nome do estudante")
    ano = st.selectbox("Ano escolar", [1, 2, 3, 4, 5], index=2)
    complexidade = st.selectbox("Complexidade estimada do texto", ["Simples", "Intermediário", "Complexo"], index=1)
    st.caption("A estimativa de tempo é orientativa e deve ser validada pelo professor.")

st.subheader("1. Texto de leitura")
uploaded = st.file_uploader("Anexar texto (TXT, DOCX ou PDF)", type=["txt", "docx", "pdf"])
texto_arquivo = extrair_texto_arquivo(uploaded) if uploaded else ""
texto = st.text_area("Ou cole / edite o texto aqui", value=texto_arquivo, height=220,
                     placeholder="Cole o texto que o estudante deverá ler...")
palavras_orig = tokenizar(texto)
n_palavras = len(palavras_orig)

if n_palavras:
    tmin, tmax = estimar_tempo(ano, n_palavras, complexidade)
    st.info(f"Texto com **{n_palavras} palavras**. Tempo estimado orientativo para o {ano}º ano: **{tmin//60} min {tmin%60:02d} s a {tmax//60} min {tmax%60:02d} s**.")
else:
    tmin = tmax = 0
    st.caption("O tempo estimado aparecerá quando o texto for inserido.")

st.subheader("2. Gravação da leitura")
audio = st.audio_input("Gravar o estudante lendo o texto completo")

st.subheader("3. Transcrição (opcional)")
transcricao_manual = st.text_area(
    "Transcrição reconhecida pela IA ou corrigida pelo professor",
    placeholder="Após transcrever o áudio, você poderá revisar ou corrigir este texto.",
    height=130
)

if st.button("🎙️ Transcrever áudio", type="secondary", disabled=(audio is None)):
    with st.spinner("Transcrevendo áudio com Whisper..."):
        suffix = Path(audio.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio.getvalue())
            caminho = tmp.name
        try:
            transcricao = transcrever_audio(caminho)
            st.session_state["transcricao_whisper"] = transcricao
            st.success("Transcrição concluída. Revise o texto abaixo antes de gerar o diagnóstico.")
            st.text_area("Transcrição obtida", value=transcricao, height=150, key="transcricao_resultado")
        except Exception as e:
            st.error(f"Falha na transcrição: {e}")
        finally:
            try:
                os.remove(caminho)
            except OSError:
                pass

transcricao = st.session_state.get("transcricao_whisper", "")
if transcricao_manual.strip():
    transcricao = transcricao_manual.strip()

st.subheader("4. Diagnóstico")
if st.button("📊 Gerar diagnóstico", type="primary"):
    if not texto.strip():
        st.error("Insira o texto original antes de avaliar.")
    elif not aluno.strip():
        st.error("Informe o nome do estudante.")
    elif not transcricao.strip():
        st.error("Transcreva o áudio ou cole a transcrição revisada pelo professor.")
    else:
        analise = comparar_textos(texto, transcricao)
        # Preferir duração real do arquivo de áudio. Se não disponível, solicitar duração.
        duracao_audio = None
        if audio is not None:
            try:
                import librosa
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio.name).suffix or ".wav") as tmp:
                    tmp.write(audio.getvalue())
                    audio_path = tmp.name
                duracao_audio = float(librosa.get_duration(path=audio_path))
                os.remove(audio_path)
            except Exception:
                duracao_audio = None

        if not duracao_audio or duracao_audio <= 0:
            duracao_audio = st.number_input("Informe a duração total da leitura em segundos", min_value=1.0, value=60.0, step=1.0)

        ppm_total = (analise["total_transcrito"] / duracao_audio * 60) if duracao_audio else 0
        ppm_corretas = (analise["corretas"] / duracao_audio * 60) if duracao_audio else 0
        nivel, meta = classificacao(ano, ppm_corretas, analise["precisao"])
        recomendacao = orientacao(nivel, analise["precisao"], ppm_corretas, ano, meta, analise["substituicoes"], analise["omissoes"])

        st.markdown(f"### Resultado: {aluno}")
        st.markdown(f"**{nivel}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duração", f"{int(duracao_audio//60)}:{int(duracao_audio%60):02d}")
        c2.metric("Palavras/minuto", f"{ppm_total:.1f}")
        c3.metric("Palavras corretas/minuto", f"{ppm_corretas:.1f}")
        c4.metric("Precisão", f"{analise['precisao']:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Substituições", analise["substituicoes"])
        c6.metric("Omissões", analise["omissoes"])
        c7.metric("Inserções", analise["insercoes"])
        c8.metric("Erros identificados", analise["erros"])

        st.warning("A transcrição automática pode errar, especialmente em crianças, sotaques, fala baixa e ruído. Confirme o resultado ouvindo a gravação antes de registrar a avaliação.")
        st.markdown("#### Orientação pedagógica")
        st.write(recomendacao)
        st.markdown("#### Comparação palavra a palavra")
        st.dataframe(pd.DataFrame(analise["linhas"]), use_container_width=True, hide_index=True)

        registro = {
            "Data/Hora": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Professor/Turma": professor,
            "Aluno": aluno,
            "Ano": ano,
            "Palavras texto": analise["total_original"],
            "Palavras transcritas": analise["total_transcrito"],
            "Duração (s)": round(duracao_audio, 2),
            "PPM": round(ppm_total, 1),
            "PCPM": round(ppm_corretas, 1),
            "Precisão (%)": round(analise["precisao"], 1),
            "Substituições": analise["substituicoes"],
            "Omissões": analise["omissoes"],
            "Inserções": analise["insercoes"],
            "Nível": nivel
        }
        salvar_historico(registro)
        st.success("Avaliação salva no histórico local.")

        html_relatorio = f"""<!doctype html><html lang="pt-br"><meta charset="utf-8">
        <title>Relatório de Fluência</title><body style="font-family:Arial;padding:24px">
        <h1>Relatório de Fluência Leitora</h1>
        <p><b>Professor/Turma:</b> {html.escape(professor)}</p>
        <p><b>Aluno:</b> {html.escape(aluno)} | <b>Ano:</b> {ano}º</p>
        <h2>Classificação: {html.escape(nivel)}</h2>
        <ul><li>Duração: {duracao_audio:.1f} s</li><li>PPM: {ppm_total:.1f}</li>
        <li>PCPM: {ppm_corretas:.1f}</li><li>Precisão: {analise['precisao']:.1f}%</li>
        <li>Substituições: {analise['substituicoes']}</li><li>Omissões: {analise['omissoes']}</li>
        <li>Inserções: {analise['insercoes']}</li></ul>
        <h3>Orientação pedagógica</h3><p>{html.escape(recomendacao)}</p>
        <p><i>Resultado automatizado para apoio ao professor. Revisar o áudio e validar pedagogicamente.</i></p>
        </body></html>"""
        st.download_button("⬇️ Baixar relatório HTML", html_relatorio, file_name=f"relatorio_{aluno.strip().replace(' ','_')}.html", mime="text/html")
        st.download_button("⬇️ Baixar histórico CSV", HISTORY_FILE.read_bytes(), file_name="historico_fluencia.csv", mime="text/csv")

st.divider()
st.caption("Aplicativo de apoio pedagógico. A classificação é provisória e deve ser validada pelo professor.")

