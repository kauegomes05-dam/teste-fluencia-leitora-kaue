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
import difflib

# Configuração da página
st.set_page_config(
    page_title="Teste de Fluência Leitora - Professor Kauê",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Teste de Fluência Leitora - Professor Kauê")

# Carregar o modelo Whisper otimizado (base) em cache
@st.cache_resource
def carregar_whisper():
    return whisper.load_model("base")

with st.spinner("A carregar o modelo de Inteligência Artificial Whisper (aguarde apenas no primeiro acesso)..."):
    modelo_whisper = carregar_whisper()

def limpar_palavra(palavra):
    return re.sub(r'[^\w\s]', '', palavra).lower().strip()

def transcrever_e_calcular(caminho_audio, texto_referencia):
    if not caminho_audio:
        return "", 0.0
    
    try:
        duracao = librosa.get_duration(path=caminho_audio)
        duracao = round(duracao, 1)
    except Exception:
        duracao = 0.0
        
    texto = ""
    try:
        # Prompt restritivo estrito em português do Brasil
        prompt_contexto = f"O aluno lê em voz alta o texto: {texto_referencia}"
        
        resultado = modelo_whisper.transcribe(
            caminho_audio, 
            language="pt", 
            task="transcribe", 
            temperature=0.0, # Zero criatividade para evitar invenções
            condition_on_previous_text=False, # Impede que o modelo entre em loop ou alucine outros idiomas
            word_timestamps=True,
            initial_prompt=prompt_contexto
        )
        texto = resultado.get("text", "").strip()
        
        # Limpar eventuais artefactos de HTML ou tags indesejadas geradas pela IA
        texto = re.sub(r'<[^>]*>', '', texto)
        
    except Exception as e:
        st.error(f"⚠️ Erro ao transcrever áudio com o Whisper: {e}")
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
    df_vazio = pd.DataFrame(columns=[
        "Data/Hora", "Professor/Turma", "Aluno", "Ano", 
        "Velocidade (PPM)", "Taxa Acertos (%)", "Indice Erros", "Prosodia", 
        "Nivel Proficiencia", "Tempo (s)", "Audio"
    ])
    return df_vazio, None

# Inicializar o estado da sessão
if "nome_professor" not in st.session_state: st.session_state.nome_professor = ""
if "nome_aluno" not in st.session_state: st.session_state.nome_aluno = ""
if "ano_escolar" not in st.session_state: st.session_state.ano_escolar = 2
if "texto_original" not in st.session_state: st.session_state.texto_original = ""
if "texto_transcrito_manual" not in st.session_state: st.session_state.texto_transcrito_manual = ""

# Criar separadores (Tabs)
aba1, aba2 = st.tabs(["📝 Nova Avaliação Individual", "📊 Painel e Histórico da Turma"])

with aba1:
    st.markdown("### Preencha os dados da avaliação")
    
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.nome_professor = st.text_input("Nome do Professor / Turma", value=st.session_state.nome_professor, placeholder="Ex: Prof. Kauê - 3º Ano A")
    with col2:
        st.session_state.nome_aluno = st.text_input("Nome Completo do Aluno", value=st.session_state.nome_aluno, placeholder="Digite o nome do estudante...")
        
    col3, col4 = st.columns(2)
    with col3:
        indices_anos = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
        idx_atual = indices_anos.get(st.session_state.ano_escolar, 2)
        escolha_ano = st.selectbox("Ano Escolar", options=[1, 2, 3, 4, 5], index=idx_atual)
        st.session_state.ano_escolar = escolha_ano
    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        
    audio_file = st.audio_input("Gravação da Leitura (Microfone)")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.session_state.texto_original = st.text_area("Texto Original (Referência)", value=st.session_state.texto_original, placeholder="Digite o título na 1ª linha e o texto nas linhas abaixo...", height=150)
    with col_t2:
        st.session_state.texto_transcrito_manual = st.text_area("Texto Lido (Transcrito por IA ou ajustável - Edite se necessário)", value=st.session_state.texto_transcrito_manual, placeholder="Deixe em branco para transcrever automaticamente pelo microfone...", height=150)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        gerar_diagnostico = st.button("🚀 Gerar Diagnóstico Multidimensional", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🔄 Refazer / Limpar Avaliação", use_container_width=True):
            st.session_state.nome_aluno = ""
            st.session_state.texto_original = ""
            st.session_state.texto_transcrito_manual = ""
            st.rerun()

    if gerar_diagnostico:
        prof_val = st.session_state.nome_professor.strip()
        aluno_val = st.session_state.nome_aluno.strip()
        orig_val = st.session_state.texto_original.strip()
        
        if not prof_val:
            prof_val = "Professor_Anonimo"
        if not aluno_val:
            aluno_val = "Aluno Anonimo"
            
        if not orig_val:
            st.warning("⚠️ Por favor, preencha o campo 'Texto Original (Referência)' antes de gerar o diagnóstico.")
        else:
            caminho_audio_temp = None
            if audio_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
                    tmp.write(audio_file.read())
                    caminho_audio_temp = tmp.name
            
            with st.spinner("A analisar a leitura com rigor e estabilidade..."):
                texto_auto, duracao_auto = transcrever_e_calcular(caminho_audio_temp, orig_val)
            
            tempo_real = duracao_auto if duracao_auto > 0 else 15.0
            
            if not st.session_state.texto_transcrito_manual.strip() and texto_auto:
                st.session_state.texto_transcrito_manual = texto_auto
                st.rerun()
                
            texto_a_usar = st.session_state.texto_transcrito_manual.strip()
            if not texto_a_usar:
                texto_a_usar = texto_auto
                
            nome_ficheiro_audio = "Sem_Audio"
            if caminho_audio_temp:
                try:
                    pasta_audios = "audios_turma"
                    os.makedirs(pasta_audios, exist_ok=True)
                    nome_limpo_aluno = re.sub(r'[^\w\s]', '', aluno_val).strip().replace(' ', '_')
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_ficheiro_audio = f"{nome_limpo_aluno}_{timestamp}.webm"
                    shutil.copy(caminho_audio_temp, os.path.join(pasta_audios, nome_ficheiro_audio))
                except Exception:
                    pass

            lista_original_raw = orig_val.split()
            lista_lida_raw = texto_a_usar.split()
            
            palavras_originais_limpas = [limpar_palavra(p) for p in lista_original_raw]
            palavras_lidas_limpas = [limpar_palavra(p) for p in lista_lida_raw]
            
            matcher = difflib.SequenceMatcher(None, palavras_originais_limpas, palavras_lidas_limpas)
            
            resultado_analise = []
            erros_detalhados = []
            acertos = 0
            total_palavras = len(palavras_originais_limpas)
            total_erros = 0
            
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == 'equal':
                    for k in range(i2 - i1):
                        orig_orig = lista_original_raw[i1 + k]
                        resultado_analise.append((orig_orig, orig_orig, "CORRETO"))
                        acertos += 1
                elif tag == 'replace':
                    for k in range(max(i2 - i1, j2 - j1)):
                        orig_orig = lista_original_raw[i1 + k] if (i1 + k) < i2 else ""
                        lido_orig = lista_lida_raw[j1 + k] if (j1 + k) < j2 else ""
                        if orig_orig and lido_orig:
                            sub_matcher = difflib.SequenceMatcher(None, limpar_palavra(orig_orig), limpar_palavra(lido_orig))
                            if sub_matcher.ratio() >= 0.5:
                                resultado_analise.append((orig_orig, lido_orig, "VARIACAO_FONETICA"))
                                acertos += 0.5
                                total_erros += 1
                                erros_detalhados.append(f"Troca fonética: [{orig_orig}] lido como [{lido_orig}]")
                            else:
                                resultado_analise.append((orig_orig, lido_orig, "ERRADO"))
                                total_erros += 1
                                erros_detalhados.append(f"Erro/Troca estrutural: Esperado [{orig_orig}] --> Lido [{lido_orig}]")
                        elif orig_orig:
                            resultado_analise.append((orig_orig, "", "OMITIDO"))
                            total_erros += 1
                            erros_detalhados.append(f"Omitido: [{orig_orig}]")
                elif tag == 'delete':
                    for k in range(i2 - i1):
                        orig_orig = lista_original_raw[i1 + k]
                        resultado_analise.append((orig_orig, "", "OMITIDO"))
                        total_erros += 1
                        erros_detalhados.append(f"Omitido: [{orig_orig}]")

            taxa_acertos = (acertos / total_palavras) * 100 if total_palavras > 0 else 0
            if taxa_acertos > 100: taxa_acertos = 100.0
            
            minutos_gastos = tempo_real / 60 if tempo_real > 0 else 1
            velocidade_ppm = round(total_palavras / minutos_gastos) if minutos_gastos > 0 else 0
            
            metas_ppm = {1: 35, 2: 60, 3: 90, 4: 110, 5: 130}
            meta_esperada = metas_ppm.get(int(st.session_state.ano_escolar), 80)
            razao_ritmo = velocidade_ppm / meta_esperada if meta_esperada > 0 else 0
            
            if tempo_real > (total_palavras * 2.5):
                prosodia_indicador = "Pausada / Fragmentada (Hesitações frequentes)"
            elif tempo_real < (total_palavras * 0.8):
                prosodia_indicador = "Acelerada / Precipitada"
            else:
                prosodia_indicador = "Fluida com continuidade e ritmo adequado"

            if taxa_acertos < 70 or razao_ritmo < 0.5:
                nivel_proficiencia = "🔴 Abaixo do básico"
                interpretacao_nivel = "Fluência ainda muito distante das expectativas para a escolaridade."
            elif taxa_acertos < 85 or razao_ritmo < 0.75:
                nivel_proficiencia = "🟠 Básico"
                interpretacao_nivel = "Apresenta habilidades iniciais de fluência, mas necessita avançar."
            elif taxa_acertos < 95 or razao_ritmo < 1.0:
                nivel_proficiencia = "🟡 Adequado"
                interpretacao_nivel = "Demonstra fluência compatível com as expectativas da escolaridade."
            else:
                nivel_proficiencia = "🟢 Avançado"
                interpretacao_nivel = "Apresenta desempenho superior às expectativas previstas."

            if velocidade_ppm < meta_esperada * 0.6 and taxa_acertos < 80:
                orientacao_pratica = "Baixa velocidade e poucos acertos: trabalhar reconhecimento de palavras, leitura repetida e consciência fonológica."
            elif velocidade_ppm >= meta_esperada * 0.8 and taxa_acertos < 85:
                orientacao_pratica = "Boa velocidade, mas muitos erros: desenvolver precisão, atenção às palavras e correspondência entre grafemas e fonemas."
            elif taxa_acertos >= 85 and "Pausada" in prosodia_indicador:
                orientacao_pratica = "Leitura correta, mas muito pausada: trabalhar continuidade, ritmo e leitura de frases."
            else:
                orientacao_pratica = "Boa fluência: ampliar o repertório textual e trabalhar leitura expressiva, inferências e compreensão."

            html_texto = "<div style='font-size: 18px; line-height: 1.8; background-color: #1e1e1e; padding: 15px; border-radius: 8px; color: #ffffff;'>"
            for orig, lido, status in resultado_analise:
                if status == "CORRETO":
                    html_texto += f"<span style='color: #4ade80; font-weight: bold;'>{orig}</span> "
                elif status == "VARIACAO_FONETICA":
                    html_texto += f"<span style='color: #fbbf24; text-decoration: underline; font-weight: bold;'>⚡{orig} ({lido})</span> "
                elif status == "ERRADO":
                    html_texto += f"<span style='color: #f87171; text-decoration: line-through; font-weight: bold;'>{orig}</span><span style='color: #f87171; font-size: 15px; font-weight: bold;'>({lido})</span> "
                else:
                    html_texto += f"<span style='color: #ef4444; text-decoration: underline; font-style: italic;'>{orig} [Omitido]</span> "
            html_texto += "</div>"
            
            st.markdown(f"### 📊 Relatório Diagnóstico: {aluno_val} ({st.session_state.ano_escolar}º Ano)")
            
            df_indicadores = pd.DataFrame({
                "Indicador": ["Velocidade", "Taxa de acertos", "Índice de erros", "Prosódia", "Nível de Proficiência"],
                "O que permite verificar / Resultado": [
                    f"{velocidade_ppm} Palavras por minuto (Meta: {meta_esperada} PPM)",
                    f"{taxa_acertos:.1f}% de acertos corretos",
                    f"{total_erros} equívocos (trocas, omissões ou falhas)",
                    f"{prosodia_indicador}",
                    f"{nivel_proficiencia} — {interpretacao_nivel}"
                ]
            })
            st.table(df_indicadores)
            
            st.markdown(f"### 💡 Orientação Pedagógica Recomendada:")
            st.info(orientacao_pratica)
            
            st.markdown("#### 🔍 Transparência (Comparação entre Texto Original e Lido):")
            st.markdown(html_texto, unsafe_allow_html=True)
            
            if caminho_audio_temp and os.path.exists(caminho_audio_temp):
                st.markdown("#### 🎧 Áudio da Leitura Realizada:")
                st.audio(caminho_audio_temp)

            ficheiro_csv = obter_nome_ficheiro_csv(prof_val)
            file_exists = os.path.exists(ficheiro_csv)
            with open(ficheiro_csv, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "Data/Hora", "Professor/Turma", "Aluno", "Ano", 
                        "Velocidade (PPM)", "Taxa Acertos (%)", "Indice Erros", "Prosodia", 
                        "Nivel Proficiencia", "Tempo (s)", "Audio"
                    ])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M"), prof_val, aluno_val, st.session_state.ano_escolar,
                    velocidade_ppm, f"{taxa_acertos:.1f}", total_erros, prosodia_indicador,
                    nivel_proficiencia, tempo_real, nome_ficheiro_audio
                ])
            
            nome_limpo_arq = re.sub(r'[^\w\s]', '', aluno_val).strip().replace(' ', '_')
            ficheiro_individual = f"relatorio_{nome_limpo_arq}.html"
            
            conteudo_html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><title>Relatório - {aluno_val}</title></head>
<body style="font-family: Arial, sans-serif; background: #f4f6f9; padding: 20px;">
    <div style="background: #fff; padding: 25px; border-radius: 10px; max-width: 800px; margin: auto;">
        <h2>📚 Teste de Fluência Leitora - Professor Kauê</h2>
        <p><strong>Professor / Turma:</strong> {prof_val}</p>
        <p><strong>Aluno:</strong> {aluno_val} ({st.session_state.ano_escolar}º Ano)</p>
        <hr>
        <h3>Indicadores de Desempenho</h3>
        <ul>
            <li><strong>Velocidade:</strong> {velocidade_ppm} PPM (Meta: {meta_esperada})</li>
            <li><strong>Taxa de Acertos:</strong> {taxa_acertos:.1f}%</li>
            <li><strong>Índice de Erros:</strong> {total_erros} ocorrências</li>
            <li><strong>Prosódia:</strong> {prosodia_indicador}</li>
            <li><strong>Nível de Proficiência:</strong> {nivel_proficiencia} ({interpretacao_nivel})</li>
        </ul>
        <h3>Orientação Pedagógica</h3>
        <p style="background: #e2e8f0; padding: 12px; border-radius: 6px;">{orientacao_pratica}</p>
        <h3>Análise Textual Detalhada</h3>
        <div style="background: #1e1e1e; padding: 15px; border-radius: 8px; color: #fff; margin-top: 15px;">{html_texto}</div>
    </div></body></html>"""
            
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
