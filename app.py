import streamlit as st
import csv
import pandas as pd
from datetime import datetime
import os
import re
import shutil
import librosa
import torch
import whisper
import tempfile

# Configuração da página
st.set_page_config(
    page_title="Teste de Fluência Leitora - Professor Kauê",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Teste de Fluência Leitora - Professor Kauê")
st.markdown("### **Ambiente Multi-Professor / Turma Isolada**")

# Carregar o modelo Whisper em cache para otimizar a memória
@st.cache_resource
def carregar_whisper():
    return whisper.load_model("small")

with st.spinner("A carregar o modelo de Inteligência Artificial Whisper (aguarde apenas no primeiro acesso)..."):
    modelo_whisper = carregar_whisper()

def limpar_palavra(palavra):
    return re.sub(r'[^\w\s]', '', palavra).lower().strip()

def transcrever_e_calcular(caminho_audio):
    if not caminho_audio:
        return "", 0.0
    
    try:
        duracao = librosa.get_duration(path=caminho_audio)
        duracao = round(duracao, 1)
    except Exception:
        duracao = 0.0
        
    texto = ""
    try:
        resultado = modelo_whisper.transcribe(
            caminho_audio, 
            language="pt", 
            task="transcribe", 
            temperature=0.0,
            initial_prompt="Avaliação de leitura escolar em português do Brasil. O aluno lê o texto em voz alta palavra por palavra."
        )
        texto = resultado.get("text", "").strip()
    except Exception:
        texto = ""
        
    return texto, duracao

def obter_nome_ficheiro_csv(nome_professor):
    if not nome_professor or not nome_professor.strip():
        nome_professor = "Professor_Geral"
    limpo = re.sub(r'[^\w\s]', '', nome_professor).strip().replace(' ', '_')
    return f"historico_{limpo}.csv"

def carregar_tabela_turma(nome_professor):
    if not nome_professor or not nome_professor.strip():
        nome_professor = "Professor_Geral"
    ficheiro_csv = obter_nome_ficheiro_csv(nome_professor)
    if os.path.exists(ficheiro_csv):
        try:
            df = pd.read_csv(ficheiro_csv)
            return df, ficheiro_csv
        except Exception:
            pass
    df_vazio = pd.DataFrame(columns=["Data/Hora", "Professor/Turma", "Aluno", "Ano", "Precisao (%)", "Ritmo Real (PPM)", "Meta (PPM)", "Nivel Desempenho", "Tempo (s)", "Texto Lido", "Audio"])
    return df_vazio, None

# Criar separadores (Tabs)
aba1, aba2 = st.tabs(["📝 Nova Avaliação Individual", "📊 Painel e Histórico da Turma"])

with aba1:
    st.markdown("### Preencha os dados da avaliação")
    
    col1, col2 = st.columns(2)
    with col1:
        nome_professor = st.text_input("Nome do Professor / Turma", value="", placeholder="Ex: Prof. Kauê - 3º Ano A")
    with col2:
        nome_aluno = st.text_input("Nome Completo do Aluno", value="", placeholder="Digite o nome do estudante...")
        
    col3, col4 = st.columns(2)
    with col3:
        ano_escolar = st.selectbox("Ano Escolar", options=[1, 2, 3, 4, 5], index=2)
    with col4:
        st.markdown("<br>", unsafe_allow_html=True) # Espaçamento
        
    audio_file = st.audio_input("Gravação da Leitura (Microfone)")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        texto_original = st.text_area("Texto Original (Referência)", placeholder="Digite o título na 1ª linha e o texto nas linhas abaixo...", height=150)
    with col_t2:
        texto_transcrito_manual = st.text_area("Texto Lido (Transcrito por IA ou ajustável)", placeholder="Deixe em branco para transcrever automaticamente pelo microfone...", height=150)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        gerar_diagnostico = st.button("🚀 Gerar Diagnóstico Individual do Aluno", type="primary", use_container_width=True)

    if gerar_diagnostico:
        if not nome_professor.strip():
            nome_professor = "Professor_Anonimo"
        if not nome_aluno.strip():
            nome_aluno = "Aluno Anonimo"
            
        if not texto_original.strip():
            st.warning("⚠️ Por favor, preencha o campo 'Texto Original (Referência)' antes de gerar o diagnóstico.")
        else:
            caminho_audio_temp = None
            if audio_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_file.read())
                    caminho_audio_temp = tmp.name
            
            texto_auto, duracao_auto = transcrever_e_calcular(caminho_audio_temp)
            tempo_real = duracao_auto if duracao_auto > 0 else 15.0
            
            texto_a_usar = texto_transcrito_manual
            if not texto_a_usar or not texto_a_usar.strip():
                texto_a_usar = texto_auto
                
            nome_ficheiro_audio = "Sem_Audio"
            if caminho_audio_temp:
                try:
                    pasta_audios = "audios_turma"
                    os.makedirs(pasta_audios, exist_ok=True)
                    nome_limpo_aluno = re.sub(r'[^\w\s]', '', nome_aluno).strip().replace(' ', '_')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_ficheiro_audio = f"{nome_limpo_aluno}_{timestamp}.wav"
                    shutil.copy(caminho_audio_temp, os.path.join(pasta_audios, nome_ficheiro_audio))
                except Exception:
                    pass

            lista_original_raw = texto_original.split()
            lista_lida_raw = texto_a_usar.split()
            
            palavras_originais_limpas = [limpar_palavra(p) for p in lista_original_raw]
            palavras_lidas_limpas = [limpar_palavra(p) for p in lista_lida_raw]
            
            resultado_analise = []
            erros_detalhados = []
            acertos = 0
            
            for i, palavra_origem_limpa in enumerate(palavras_originais_limpas):
                palavra_original_original = lista_original_raw[i]
                if i < len(palavras_lidas_limpas):
                    palavra_falada_limpa = palavras_lidas_limpas[i]
                    palavra_falada_original = lista_lida_raw[i] if i < len(lista_lida_raw) else ""
                    
                    if palavra_falada_limpa == palavra_origem_limpa:
                        resultado_analise.append((palavra_original_original, palavra_falada_original, "CORRETO"))
                        acertos += 1
                    else:
                        resultado_analise.append((palavra_original_original, palavra_falada_original, "ERRADO"))
                        erros_detalhados.append(f"Esperado: [{palavra_original_original}] --> Lido: [{palavra_falada_original}]")
                else:
                    resultado_analise.append((palavra_original_original, "", "OMITIDO"))
                    erros_detalhados.append(f"Esperado: [{palavra_original_original}] --> Omitido")
                    
            total_palavras = len(palavras_originais_limpas)
            precisao = (acertos / total_palavras) * 100 if total_palavras > 0 else 0
            
            metas_ppm = {1: 35, 2: 60, 3: 90, 4: 110, 5: 130}
            ppm_esperado = metas_ppm.get(int(ano_escolar), 80)
            
            minutos_gastos = tempo_real / 60 if tempo_real > 0 else 1
            ppm_real = total_palavras / minutos_gastos if minutos_gastos > 0 else 0
            razao_ritmo = ppm_real / ppm_esperado if ppm_esperado > 0 else 0
            
            if precisao < 70 or razao_ritmo < 0.5:
                nivel_desempenho = "🔴 Abaixo do Básico: Dificuldades importantes no reconhecimento de palavras e ritmo."
            elif precisao < 85 or razao_ritmo < 0.75:
                nivel_desempenho = "🟠 Básico: Leitura inicial ou com tropeços, domínio parcial da fluência."
            elif precisao < 95 or razao_ritmo < 1.0:
                nivel_desempenho = "🟡 Adequado: Boa precisão e velocidade compatível com a etapa."
            else:
                nivel_desempenho = "🟢 Avançado: Excelente fluência, ritmo adequado e alto índice de acertos."

            html_texto = "<div style='font-size: 18px; line-height: 1.8; background-color: #1e1e1e; padding: 15px; border-radius: 8px; color: #ffffff;'>"
            for orig, lido, status in resultado_analise:
                if status == "CORRETO":
                    html_texto += f"<span style='color: #4ade80; font-weight: bold;'>{orig}</span> "
                elif status == "ERRADO":
                    html_texto += f"<span style='color: #f87171; text-decoration: line-through; font-weight: bold;'>{orig}</span><span style='color: #f87171; font-size: 15px; font-weight: bold;'>({lido})</span> "
                else:
                    html_texto += f"<span style='color: #fbbf24; text-decoration: underline; font-style: italic;'>{orig} [Omitido]</span> "
            html_texto += "</div>"
            
            st.markdown(f"### 📊 Relatório Diagnóstico: {nome_aluno} ({ano_escolar}º Ano)")
            st.markdown(f"- **Precisão de Leitura:** {precisao:.1f}%")
            st.markdown(f"- **Ritmo Real:** {round(ppm_real)} PPM (Meta esperada: {ppm_esperado} PPM)")
            st.markdown(f"- **Tempo Gasto:** {tempo_real} segundos")
            st.markdown(f"- **Nível:** {nivel_desempenho}")
            
            st.markdown("#### 🔍 Análise Visual Detalhada:")
            st.markdown(html_texto, unsafe_allow_html=True)
            
            # Guardar histórico CSV
            ficheiro_csv = obter_nome_ficheiro_csv(nome_professor)
            file_exists = os.path.exists(ficheiro_csv)
            with open(ficheiro_csv, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Data/Hora", "Professor/Turma", "Aluno", "Ano", "Precisao (%)", "Ritmo Real (PPM)", "Meta (PPM)", "Nivel Desempenho", "Tempo (s)", "Texto Lido", "Audio"])
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), nome_professor, nome_aluno, ano_escolar, f"{precisao:.1f}", round(ppm_real), ppm_esperado, nivel_desempenho, tempo_real, texto_a_usar, nome_ficheiro_audio])
            
            # Gerar HTML individual para download
            nome_limpo_arq = re.sub(r'[^\w\s]', '', nome_aluno).strip().replace(' ', '_')
            ficheiro_individual = f"relatorio_{nome_limpo_arq}.html"
            texto_erros_formatado = "<br>".join(erros_detalhados) if erros_detalhados else "Nenhum erro detetado (100% de precisão)."
            
            conteudo_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>Relatório - {nome_aluno}</title></head>
<body style="font-family: Arial, sans-serif; background: #f4f6f9; padding: 20px;">
    <div style="background: #fff; padding: 25px; border-radius: 10px; max-width: 800px; margin: auto;">
        <h2>📚 Teste de Fluência Leitora - Professor Kauê</h2>
        <p><strong>Professor / Turma:</strong> {nome_professor}</p>
        <p><strong>Aluno:</strong> {nome_aluno} ({ano_escolar}º Ano)</p>
        <p><strong>Precisão:</strong> {precisao:.1f}% | <strong>Ritmo:</strong> {round(ppm_real)} PPM</p>
        <div style="background: #1e1e1e; padding: 15px; border-radius: 8px; color: #fff; margin-top: 15px;">{html_texto}</div>
    </div>
</body></html>"""
            
            with open(ficheiro_individual, "w", encoding="utf-8") as f_ind:
                f_ind.write(conteudo_html)
                
            with open(ficheiro_individual, "rb") as f_dl:
                st.download_button("📥 Descarregar Relatório Individual (.html)", f_dl, file_name=ficheiro_individual, mime="text/html")

with aba2:
    st.markdown("### 📊 Painel e Histórico da Turma")
    st.markdown("Insira o nome do professor ou turma para consultar os registos e descarregar a base de dados exclusiva.")
    
    input_busca_prof = st.text_input("Consultar Professor / Turma", value="", placeholder="Digite exatamente o nome usado...")
    
    if st.button("🔍 Carregar Histórico"):
        df_turma, f_csv = carregar_tabela_turma(input_busca_prof)
        if not df_turma.empty:
            st.dataframe(df_turma, use_container_width=True)
            if f_csv and os.path.exists(f_csv):
                with open(f_csv, "rb") as f_csv_dl:
                    st.download_button("📥 Descarregar Base Completa desta Turma (CSV)", f_csv_dl, file_name=f_csv, mime="text/csv")
        else:
            st.info("Ainda não existem registos guardados para este nome de professor/turma.")
