#!/bin/bash

# Diretórios e caminhos
INPUT_FILE="data/golden_v2.json"
OUTPUT_DIR="data/samples"
MODEL="models/pucpr-br_tempclin-biobertpt-all"

# Cria a pasta de saída, caso não exista
mkdir -p "$OUTPUT_DIR"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Erro: Arquivo $INPUT_FILE não encontrado. Execute o script na raiz do projeto."
    exit 1
fi

echo "Lendo $INPUT_FILE e extraindo amostras..."

# Processa o JSON com jq e gera os arquivos
jq -c "
.records[] | 
select(
    .flow_input.evolucao_medica != null and .flow_input.evolucao_medica != \"\" and 
    .flow_input.evolucao_enfermagem != null and .flow_input.evolucao_enfermagem != \"\" and 
    .flow_input.prescricao != null and .flow_input.prescricao != \"\" and 
    .flow_input.resumo_da_internacao != null and .flow_input.resumo_da_internacao != \"\"
) | 
{
    id: .sample_id,
    content: {
        model: \"$MODEL\",
        input: [
            (\"evolucao_medica: \" + .flow_input.evolucao_medica),
            (\"evolucao_enfermagem: \" + .flow_input.evolucao_enfermagem),
            (\"prescricao: \" + .flow_input.prescricao),
            (\"resumo_da_internacao: \" + .flow_input.resumo_da_internacao)
        ]
    }
}
" "$INPUT_FILE" | while IFS= read -r line; do
    # Extrai o sample_id e o conteúdo para salvar no arquivo correspondente
    id=$(echo "$line" | jq -r '.id')
    echo "$line" | jq '.content' > "${OUTPUT_DIR}/${id}.json"
    echo "Criado: ${OUTPUT_DIR}/${id}.json"
done

COUNT=$(ls -1q "$OUTPUT_DIR"/*.json | wc -l)
echo "Processo concluído. $COUNT arquivos criados em $OUTPUT_DIR."
