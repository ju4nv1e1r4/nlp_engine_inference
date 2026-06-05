#!/bin/bash

SAMPLES_DIR="data/samples"
RESULTS_DIR="data/results"
ENGINE="./target/release/inference-engine"

# Cria a pasta de resultados
mkdir -p "$RESULTS_DIR"

if [ ! -f "$ENGINE" ]; then
    echo "Erro: Engine $ENGINE não encontrado na raiz do projeto."
    exit 1
fi

echo "Iniciando inferência em lote..."

# Arquivo de log combinado
MASTER_LOG="$RESULTS_DIR/all_results.log"
> "$MASTER_LOG"

count=0
total=$(ls -1q "$SAMPLES_DIR"/*.json 2>/dev/null | wc -l)

if [ "$total" -eq 0 ]; then
    echo "Nenhuma amostra encontrada em $SAMPLES_DIR."
    exit 1
fi

for sample in "$SAMPLES_DIR"/*.json; do
    filename=$(basename "$sample")
    sample_id="${filename%.*}"
    
    count=$((count + 1))
    echo "[$count/$total] Processando sample $sample_id..."
    
    echo "=== Sample: $sample_id ===" >> "$MASTER_LOG"
    
    RESULT_FILE="$RESULTS_DIR/${sample_id}_result.txt"
    
    # Verifica se o resultado já existe e não está vazio (cache)
    if [ -s "$RESULT_FILE" ]; then
        echo "  -> Usando cache (já inferido)."
    else
        # Executa a inferência e salva no arquivo individual
        $ENGINE --input "$sample" > "$RESULT_FILE" 2>&1
    fi
    
    # Anexa a saída (nova ou do cache) ao arquivo de log principal
    cat "$RESULT_FILE" >> "$MASTER_LOG"
    echo -e "\n\n" >> "$MASTER_LOG"
done

echo "Inferência concluída!"
echo "Os resultados individuais estão na pasta: $RESULTS_DIR"
echo "Um log com todos os resultados combinados foi salvo em: $MASTER_LOG"
