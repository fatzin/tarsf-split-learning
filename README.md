# Split Learning

Este repositório contém o código necessário para configurar e executar um servidor e múltiplos clientes para treinamento utilizando Split Learning.

## Como rodar

### Rode os comandos em terminais separados

#### Passo 1

```
cd split-learning
```

_**Rode em terminais separados**_

```powershell
1. python server.py
2. python client1.py
3. python client[i].py
```

- Você pode escolher o número de clientes que participam do treinamento. Neste repositório, três clientes são apresentados. Se quiser aumentar o número de clientes, duplique o programa do cliente e ajuste o número de clientes esperado no server.py.
- Além disso, você pode ajustar o número de épocas de treinamento e o learning rate nos arquivos de configuração correspondentes.

#### Passo 2

Após o término do treinamento o evaluation do modelo pode ser feito através da aplicação web escrita utilizando o Framework Flask.

```powershell
cd web
python app.py
```

e acessando a URL

```powershell
localhost:7500
```

## Aprendizado e futuro

Durante a disciplina de Tópicos Avançados em Redes sem Fio (TARSF), aprendi muito sobre redes, redes sem fio e o quão vastas são as possibilidades nessa área de conhecimento. O propósito deste trabalho era o aprendizado de um método (Split Learning) aliado a conceitos de redes, e futuramente integrar este método ao simulador de rede NS3 para manipular dados de dispositivos IoT através da conexão entre eles e um servidor, coletando parâmetros como throughput, delay, e eficiência energética. Infelizmente, com o término da disciplina, não foi possível chegar a esse nível de abstração.

## Conclusões

O Split Learning é um novo recurso no campo da Inteligência Artificial que pode ser utilizado junto ao Federated Learning para avanços na área de redes na Internet das Coisas, conforme descrito nas seguintes referências:

- [Improving the Communication and Computation Efficiency of Split Learning for IoT Applications](https://ieeexplore.ieee.org/abstract/document/9685493/references#references)
- [Combined Federated and Split Learning in Edge Computing for Ubiquitous Intelligence in Internet of Things: State of the Art and Future Directions](https://arxiv.org/abs/2207.09611)
- [DFL: Dynamic Federated Split Learning in Heterogeneous IoT](https://ieeexplore.ieee.org/document/10547401)
