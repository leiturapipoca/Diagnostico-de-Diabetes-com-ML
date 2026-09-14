# Diagnostico-de-Diabetes-com-ML

# Classificação Clássica e MLP com PyTorch — Diagnóstico de Diabetes
 
Projeto acadêmico de aprendizado de máquina aplicado ao diagnóstico de diabetes, comparando modelos clássicos de classificação (KNN e Regressão Logística) com redes neurais Multi-Layer Perceptron (MLP) implementadas em PyTorch.
 
**Autor:** Guilherme Silva Cavalcante
 
---
 
## 1. Visão Geral
 
Este notebook aborda um problema de **classificação binária**: prever a presença ou ausência de diabetes em pacientes com base em variáveis clínicas e fisiológicas. O trabalho está dividido em três etapas principais:
 
1. **Preparação e limpeza dos dados**, incluindo análise exploratória via matriz de correlação.
2. **Comparação de modelos clássicos** — `KNeighborsClassifier` e `LogisticRegression` (scikit-learn) — com otimização de hiperparâmetros via `GridSearchCV`.
3. **Construção e avaliação de redes neurais MLP** em PyTorch, testando diferentes arquiteturas, com *early stopping* baseado na *loss* de validação.
O notebook também inclui uma seção de perguntas e respostas (QA) sobre conceitos fundamentais de aprendizado de máquina, como desbalanceamento de classes, overfitting, vanishing gradient e trade-offs entre KNN e MLP.
 
## 2. Conjunto de Dados
 
O projeto utiliza o **Pima Indians Diabetes Dataset**, originalmente coletado pelo *National Institute of Diabetes and Digestive and Kidney Diseases* (EUA), contendo registros de mulheres com pelo menos 21 anos da população Pima.
 
- **Fonte utilizada no notebook:** carregada diretamente via URL pública do GitHub (`pcbrom/perceptron-mlp-cnn`), não sendo necessário download manual.
- **Referência do dataset original:** [Pima Indians Diabetes Database (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
### Variáveis de entrada
 
| Variável | Descrição |
|---|---|
| Pregnancies | Número de gestações |
| Glucose | Concentração de glicose plasmática (teste de tolerância oral) |
| BloodPressure | Pressão arterial diastólica (mm Hg) |
| SkinThickness | Espessura da dobra cutânea do tríceps (mm) |
| Insulin | Insulina sérica de 2 horas (μU/mL) |
| BMI | Índice de massa corporal (kg/m²) |
| DiabetesPedigreeFunction | Função de histórico familiar de diabetes |
| Age | Idade (anos) |
| **Outcome** | Variável alvo (0 = não diabético, 1 = diabético) |
 
## 3. Estrutura do Notebook
 
| Seção | Conteúdo |
|---|---|
| Introdução | Contexto do problema e do dataset |
| Exercício 1 | Limpeza de dados (tratamento de zeros inválidos) e matriz de correlação |
| Exercício 2 | KNN vs. Regressão Logística, com `GridSearchCV` e divisão treino/teste 80/20 |
| Exercício 3 | Três arquiteturas de MLP em PyTorch, com early stopping e divisão treino/validação/teste 70/15/15 |
| QA | Discussão teórica sobre métricas, overfitting, vanishing gradient e comparação KNN vs. MLP |
 
## 4. Premissas e Decisões Metodológicas
 
- **Tratamento de dados ausentes:** valores fisiologicamente inválidos (zeros) nas colunas `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin` e `BMI` são convertidos para `NaN` e substituídos pela **mediana** da respectiva coluna, preservando o volume de dados.
- **Padronização:** todos os modelos (KNN, Regressão Logística e MLP) utilizam `StandardScaler` para normalizar as features antes do treinamento.
- **Reprodutibilidade:** `random_state=42` é utilizado em todas as divisões de dados.
- **Estratificação:** as divisões treino/teste (e treino/validação/teste) são estratificadas para preservar a proporção original das classes.
- **Divisão de dados:**
  - Exercício 2 (modelos clássicos): 80% treino / 20% teste.
  - Exercício 3 (MLP): 70% treino / 15% validação / 15% teste.
- **Seleção de hiperparâmetros:** realizada exclusivamente sobre o conjunto de treino via validação cruzada (`GridSearchCV`, `cv=5`), evitando vazamento de dados (*data leakage*) proveniente do conjunto de teste.
- **Critério de parada do treinamento (MLP):** *early stopping* com base na loss de validação, interrompendo o treino quando não há melhora por um número definido de épocas consecutivas (*patience*), preservando os pesos do melhor modelo observado.
- **Desbalanceamento de classes:** o dataset apresenta desbalanceamento (~65% não-diabéticos vs. ~35% diabéticos); por isso, além da acurácia, métricas como F1-Score e Recall são utilizadas na avaliação dos modelos.
## 5. Tecnologias e Bibliotecas Utilizadas
 
| Biblioteca | Finalidade |
|---|---|
| `pandas` | Manipulação e análise de dados |
| `numpy` | Operações numéricas |
| `matplotlib` / `seaborn` | Visualização de dados (matriz de correlação, matrizes de confusão, curvas de aprendizado) |
| `scikit-learn` | Modelos clássicos (KNN, Regressão Logística), pré-processamento, métricas e validação cruzada |
| `torch` (PyTorch) | Construção e treinamento das redes neurais MLP |
| `copy` | Cópia dos pesos do melhor modelo durante o early stopping |
 
## 6. Como Executar
 
### Opção A — Google Colab (recomendado)
 
1. Acesse [Google Colab](https://colab.research.google.com/).
2. Faça upload do arquivo `Atividade_KNN_LR_MLP_com_PyTorch.ipynb` (**Arquivo → Fazer upload de notebook**).
3. Execute as células em ordem sequencial (**Ambiente de execução → Executar tudo**).
4. Nenhuma instalação adicional é necessária — o Colab já possui `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` e `torch` pré-instalados.
5. O dataset é carregado automaticamente via URL pública; não é necessário upload manual de arquivos.
### Opção B — Ambiente local (Jupyter)
 
**Pré-requisitos:**
- Python 3.9 ou superior
- Acesso à internet (para download do dataset via URL)
**Passos:**
 
```bash
# 1. Criar e ativar um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
 
# 2. Instalar as dependências
pip install pandas numpy matplotlib seaborn scikit-learn torch jupyter
 
# 3. Iniciar o Jupyter Notebook
jupyter notebook
```
 
Em seguida, abra o arquivo `Atividade_KNN_LR_MLP_com_PyTorch.ipynb` na interface do Jupyter e execute as células em ordem sequencial (célula por célula ou via **Cell → Run All**).
 
> **Observação:** as células devem ser executadas em ordem, pois etapas posteriores (como a construção das MLPs) dependem de variáveis e do DataFrame tratado nas células anteriores (limpeza de dados do Exercício 1).
 
## 7. Resumo dos Resultados
 
### Modelos clássicos (Exercício 2)
 
| Modelo | Conjunto | Acurácia | F1-Score |
|---|---|---:|---:|
| KNN | Treino | 0.8029 | 0.6937 |
| KNN | Teste | 0.7468 | 0.6061 |
| Regressão Logística | Treino | 0.7964 | 0.6702 |
| Regressão Logística | Teste | 0.7078 | 0.5455 |
 
### Arquiteturas MLP (Exercício 3)
 
| Arquitetura | Camadas | Neurônios | Ativação | Época de Parada | Acurácia (Teste) | F1-Score (Teste) |
|---|---|---|---|:---:|:---:|:---:|
| MLP 1 | 1 | [32] | ReLU | 165 | 0.7500 | 0.5714 |
| MLP 2 | 2 | [64, 32] | ReLU | 57 | 0.7500 | 0.5915 |
| MLP 3 | 3 | [128, 64, 32] | Tanh | 45 | 0.7672 | 0.7500 |
 
As análises detalhadas de cada resultado, incluindo interpretação estatística e discussão sobre overfitting, generalização e trade-offs entre os modelos, estão documentadas nas células de resposta do próprio notebook.
 
## 8. Estrutura de Arquivos
 
```
.
├── Atividade_KNN_LR_MLP_com_PyTorch.ipynb   # Notebook principal do projeto
└── README.md                                 # Este arquivo
```
 
## 9. Licença e Uso
 
Este projeto tem finalidade **acadêmica/educacional**. O dataset utilizado (Pima Indians Diabetes Database) é de domínio público, disponibilizado originalmente pelo National Institute of Diabetes and Digestive and Kidney Diseases
