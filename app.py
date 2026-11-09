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
    # Modelo small costuma reconhecer melhor a fala que o modelo base.
    # Na primeira execução, o Whisper fará o download do modelo.
    return whisper.load_model("small")

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
    """Transcreve automaticamente e devolve indicadores técnicos do reconhecimento."""
    model = carregar_whisper()
    result = model.transcribe(
        caminho,
        language="pt",
        task="transcribe",
        temperature=0,
        beam_size=5,
        condition_on_previous_text=False
        # Não usar o texto original como prompt para não induzir palavras na transcrição.
    )
    segmentos = result.get("segments", [])
    logprobs = [seg.get("avg_logprob") for seg in segmentos if seg.get("avg_logprob") is not None]
    silencios = [seg.get("no_speech_prob") for seg in segmentos if seg.get("no_speech_prob") is not None]
    media_logprob = sum(logprobs) / len(logprobs) if logprobs else None
    media_silencio = sum(silencios) / len(silencios) if silencios else None
    if media_logprob is None:
        confiabilidade = "Baixa"
    elif media_logprob >= -0.45 and (media_silencio is None or media_silencio < 0.35):
        confiabilidade = "Alta"
    elif media_logprob >= -0.85 and (media_silencio is None or media_silencio < 0.60):
        confiabilidade = "Moderada"
    else:
        confiabilidade = "Baixa"
    return {
        "texto": result.get("text", "").strip(),
        "confiabilidade": confiabilidade,
        "media_logprob": media_logprob,
        "media_silencio": media_silencio,
    }

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

def estimar_tempo(ano, palavras):
    # Estimativa orientativa baseada no ano escolar e extensão do texto.
    # Os intervalos são parâmetros provisórios, configuráveis pelo professor/rede.
    faixas_ppm = {
        1: (20, 40),
        2: (40, 60),
        3: (60, 90),
        4: (80, 110),
        5: (100, 130)
    }
    min_ppm, max_ppm = faixas_ppm[ano]
    tempo_min = palavras / max_ppm * 60
    tempo_max = palavras / min_ppm * 60
    return round(tempo_min), round(tempo_max)

def classificacao(ano, ppm, precisao):
    # Parâmetros iniciais por ano escolar. Precisam ser calibrados com avaliações reais.
    metas = {1: 35, 2: 60, 3: 90, 4: 110, 5: 130}
    meta = metas[ano]
    razao = ppm / meta if meta else 0
    if precisao < 70 or razao < 0.50:
        nivel = "Abaixo do Básico"
    elif precisao < 85 or razao < 0.75:
        nivel = "Básico"
    elif precisao < 95 or razao < 1.00:
        nivel = "Adequado"
    else:
        nivel = "Avançado"
    return nivel, meta


def descricao_nivel(nivel):
    descricoes = {
        "Abaixo do Básico": "Apresenta dificuldades importantes na leitura, com baixa velocidade e/ou muitos desvios em relação ao texto, indicando necessidade de apoio sistemático.",
        "Básico": "Demonstra habilidades iniciais de leitura, mas ainda apresenta limitações de precisão e/ou ritmo, necessitando de oportunidades frequentes de prática orientada.",
        "Adequado": "Apresenta desempenho compatível com os parâmetros iniciais definidos para o ano escolar, com precisão e velocidade satisfatórias.",
        "Avançado": "Apresenta desempenho acima dos parâmetros iniciais definidos para o ano escolar, com boa velocidade e alta precisão."
    }
    return descricoes.get(nivel, "Resultado da análise automática.")

def orientacao(nivel, precisao, ppm, ano, meta, substituicoes, omissoes):
    if precisao < 85:
        return "Priorizar a precisão: propor leitura compartilhada, reconhecimento de palavras, atividades de correspondência entre letras e sons e releitura de textos curtos."
    if ppm < meta * 0.75:
        return "Trabalhar a continuidade e a automatização da leitura com textos adequados ao ano escolar, leitura modelada e releituras frequentes, respeitando o ritmo do estudante."
    if substituicoes + omissoes > 0:
        return "Fortalecer a atenção às palavras e a leitura cuidadosa, com atividades de leitura em voz alta, acompanhamento de linhas e releitura de pequenos trechos."
    return "Ampliar o repertório textual, a leitura expressiva, o vocabulário e as atividades de compreensão, mantendo desafios progressivos."

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
    st.caption("A estimativa de tempo considera o ano escolar e a quantidade de palavras do texto.")

st.subheader("1. Texto de leitura")
uploaded = st.file_uploader("Anexar texto (TXT, DOCX ou PDF)", type=["txt", "docx", "pdf"])
texto_arquivo = extrair_texto_arquivo(uploaded) if uploaded else ""
texto = st.text_area("Ou cole / edite o texto aqui", value=texto_arquivo, height=220,
                     placeholder="Cole o texto que o estudante deverá ler...")
palavras_orig = tokenizar(texto)
n_palavras = len(palavras_orig)

if n_palavras:
    tmin, tmax = estimar_tempo(ano, n_palavras)
    st.info(f"Texto com **{n_palavras} palavras**. Tempo estimado orientativo para o {ano}º ano: **{tmin//60} min {tmin%60:02d} s a {tmax//60} min {tmax%60:02d} s**.")
else:
    tmin = tmax = 0
    st.caption("O tempo estimado aparecerá quando o texto for inserido.")

st.subheader("2. Gravação da leitura")
audio = st.audio_input("Gravar o estudante lendo o texto completo")
st.caption("Depois da gravação, o aplicativo transcreve e analisa a leitura automaticamente. O professor não precisa revisar nem editar a transcrição.")

st.subheader("3. Diagnóstico automático")
if st.button("📊 Analisar leitura e gerar diagnóstico", type="primary", disabled=(audio is None or not texto.strip() or not aluno.strip())):
    caminho = None
    try:
        with st.spinner("A IA está analisando a gravação. Isso pode levar alguns instantes..."):
            suffix = Path(audio.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio.getvalue())
                caminho = tmp.name

            resultado_ia = transcrever_audio(caminho)
            transcricao = resultado_ia["texto"]
            if not transcricao:
                st.error("Não foi possível reconhecer a leitura. Faça uma nova gravação em local silencioso, aproximando o dispositivo do estudante.")
                st.stop()

            import librosa
            duracao_audio = float(librosa.get_duration(path=caminho))
            if duracao_audio <= 0:
                st.error("Não foi possível calcular a duração da gravação. Faça uma nova gravação.")
                st.stop()

            analise = comparar_textos(texto, transcricao)
            ppm_total = (analise["total_transcrito"] / duracao_audio * 60) if duracao_audio else 0
            ppm_corretas = (analise["corretas"] / duracao_audio * 60) if duracao_audio else 0
            nivel, meta = classificacao(ano, ppm_corretas, analise["precisao"])
            recomendacao = orientacao(nivel, analise["precisao"], ppm_corretas, ano, meta, analise["substituicoes"], analise["omissoes"])
            descricao = descricao_nivel(nivel)

        st.markdown(f"### Resultado: {aluno}")
        st.markdown(f"## {nivel}")
        st.write(descricao)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Duração", f"{int(duracao_audio//60)}:{int(duracao_audio%60):02d}")
        c2.metric("Palavras por minuto", f"{ppm_total:.1f}")
        c3.metric("Palavras corretas por minuto", f"{ppm_corretas:.1f}")
        c4.metric("Precisão estimada", f"{analise['precisao']:.1f}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Substituições estimadas", analise["substituicoes"])
        c6.metric("Omissões estimadas", analise["omissoes"])
        c7.metric("Inserções estimadas", analise["insercoes"])
        c8.metric("Desvios identificados", analise["erros"])

        st.markdown("#### Confiabilidade do reconhecimento de voz")
        st.write(f"**{resultado_ia['confiabilidade']}**")
        if resultado_ia["confiabilidade"] == "Baixa":
            st.warning("O áudio apresentou sinais de baixa confiabilidade no reconhecimento automático. Os indicadores de precisão e os desvios podem conter falhas de transcrição; interprete a classificação com cautela.")
        elif resultado_ia["confiabilidade"] == "Moderada":
            st.info("O reconhecimento apresentou confiabilidade moderada. Pequenas diferenças entre a fala e a transcrição automática podem ocorrer.")
        else:
            st.caption("O reconhecimento apresentou bons indicadores técnicos. Ainda assim, a transcrição automática não é infalível.")

        st.markdown("#### Orientação pedagógica")
        st.write(recomendacao)
        st.caption("A classificação considera o ano escolar, a velocidade de palavras corretas por minuto e a precisão estimada. A análise não mede diretamente compreensão leitora nem prosódia/entonação.")

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
            "Precisão estimada (%)": round(analise["precisao"], 1),
            "Substituições estimadas": analise["substituicoes"],
            "Omissões estimadas": analise["omissoes"],
            "Inserções estimadas": analise["insercoes"],
            "Confiabilidade do reconhecimento": resultado_ia["confiabilidade"],
            "Nível": nivel
        }
        salvar_historico(registro)
        st.success("Diagnóstico gerado e avaliação salva no histórico local.")

        html_relatorio = f"""<!doctype html><html lang="pt-br"><meta charset="utf-8">
        <title>Relatório de Fluência</title><body style="font-family:Arial;padding:24px">
        <h1>Relatório de Fluência Leitora</h1>
        <p><b>Professor/Turma:</b> {html.escape(professor)}</p>
        <p><b>Aluno:</b> {html.escape(aluno)} | <b>Ano:</b> {ano}º</p>
        <h2>Classificação: {html.escape(nivel)}</h2><p>{html.escape(descricao)}</p>
        <ul><li>Duração: {duracao_audio:.1f} s</li><li>Palavras por minuto: {ppm_total:.1f}</li>
        <li>Palavras corretas por minuto: {ppm_corretas:.1f}</li><li>Precisão estimada: {analise['precisao']:.1f}%</li>
        <li>Substituições estimadas: {analise['substituicoes']}</li><li>Omissões estimadas: {analise['omissoes']}</li>
        <li>Inserções estimadas: {analise['insercoes']}</li><li>Confiabilidade do reconhecimento: {html.escape(resultado_ia['confiabilidade'])}</li></ul>
        <h3>Orientação pedagógica</h3><p>{html.escape(recomendacao)}</p>
        <p><i>Diagnóstico automatizado de apoio pedagógico. A precisão depende da qualidade do áudio e do reconhecimento de voz.</i></p>
        </body></html>"""
        nome_arquivo = re.sub(r"[^A-Za-z0-9_-]+", "_", aluno.strip()) or "estudante"
        st.download_button("⬇️ Baixar relatório HTML", html_relatorio, file_name=f"relatorio_{nome_arquivo}.html", mime="text/html")
        st.download_button("⬇️ Baixar histórico CSV", HISTORY_FILE.read_bytes(), file_name="historico_fluencia.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Não foi possível concluir a análise automática: {e}")
        st.info("Verifique se o áudio está audível e tente gravar novamente em um ambiente silencioso.")
    finally:
        if caminho and os.path.exists(caminho):
            try:
                os.remove(caminho)
            except OSError:
                pass

st.divider()
st.caption("Aplicativo de apoio pedagógico. Os parâmetros de classificação são iniciais e precisam ser calibrados com dados de leitura por ano escolar. A transcrição automática pode apresentar falhas.")
