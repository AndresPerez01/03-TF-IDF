# 03-TF-IDF: Motor de Búsqueda con Modelo Vectorial

###Autor: Pérez Pineda Andrés Alejandro

## El Concepto Central: Los Dos Universos Paralelos

El código no mezcla los datos. Se divide estratégicamente en dos fases o "universos espaciales" independientes para demostrar el funcionamiento del algoritmo:

1. **El Entorno de Pruebas (Corpus de Turismo):** Un corpus pequeño y controlado para verificar la correcta instanciación del modelo y limpieza de texto.
2. **El Desafío Real (Corpus Gutenberg):** Un escenario de *Big Data* donde el sistema descarga, procesa y rankea 1000 libros reales en inglés.


## Arquitectura y Flujo del Código

### Paso 1: El Entorno de Pruebas (Corpus de Turismo)

Lee un archivo local pequeño con oraciones sobre turismo en Ecuador. Aquí se instacia la primera "fábrica de vectores" (`TfidfVectorizer`).

```python
vectorizer_turismo = TfidfVectorizer()
matriz_tfidf_turismo = vectorizer_turismo.fit_transform(corpus_turismo)
```

Al ejecutar `fit_transform()`, el modelo aprende un diccionario pequeño (116 palabras) y crea una matriz matemática. Esta matriz aplica la fórmula TF-IDF, calculando la frecuencia de una palabra en una oración (TF) y multiplicándola por su rareza en todo el documento (IDF). Tras esto, este "universo" se cierra y no interfiere con el resto del código.


### Paso 2: Extracción de Datos Masivos (Web Scraping)

Para crear el corpus principal, el script se conecta automatizadamente a la biblioteca digital **Project Gutenberg** utilizando la librería `requests`. Descarga 1000 libros en formato de texto plano (`.txt`) y los almacena en el disco duro local, gestionando pausas para no saturar el servidor.


### Paso 3: Construcción del Espacio Vectorial Principal

Se crea una nueva fábrica de vectores, totalmente en blanco, para procesar los 1000 libros.

```python
vectorizer_gutenberg = TfidfVectorizer(stop_words='english', max_features=10000)
matriz_tfidf_gutenberg = vectorizer_gutenberg.fit_transform(corpus_gutenberg)
```

Al usar nuevamente `fit_transform()`, el modelo establece las reglas inmutables de este nuevo universo:

- **Las Dimensiones:** Crea un espacio geométrico de exactamente 10,000 dimensiones (las 10,000 palabras más importantes encontradas).
- **Los Pesos Históricos:** Calcula el valor IDF global de cada palabra basándose en los 1000 libros mediante la ecuación:

$$idf_{t} = \log\left(\frac{N}{df_{t}}\right)$$

- **El Mapa:** Ubica cada libro como una flecha estática ($\vec{d}$) dentro de esta matriz de 1000 filas por 10,000 columnas.



### Paso 4: El Motor de Búsqueda y la Similitud del Coseno

Para buscar información (ej. `"science fiction adventure"`), se debe introducir la consulta del usuario al universo que se creó en el Paso 3.

```python
# OJO: Se usa transform(), no fit_transform()
vector_consulta = vectorizer.transform([consulta])

# Cálculo de ángulos
similitudes = cosine_similarity(vector_consulta, matriz_tfidf).flatten()
```

> **Cambio Principal** Aquí usamos estrictamente `transform()`. Esto obliga a la nueva consulta a empaquetarse y obedecer las dimensiones y pesos históricos que los libros ya establecieron. Si usáramos `fit_transform()` por error, el sistema olvidaría los libros, crearía un universo de solo 3 dimensiones basado en la consulta y el programa colapsaría matemáticamente.

Con la consulta convertida en un vector ($\vec{q}$), rankeamos los resultados usando la **Similitud del Coseno**:

$$sim(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{||\vec{q}|| \cdot ||\vec{d}||}$$

En lugar de contar palabras repetidas, el sistema mide el ángulo entre la flecha de tu búsqueda y las flechas de los 1000 libros. Si un libro trata sobre esos temas, su flecha apuntará en la misma dirección, logrando un puntaje de similitud más cercano a `1`.

### La Normalización y la Longitud del Documento
 
En una versión anterior, el código leía solo los primeros 5,000 caracteres de cada libro. Eso significaba comparar una consulta de 3 palabras contra "libritos" de apenas ~800 palabras. Utilizando la línea de código `f.read(5000)`. 
 
Al cambiar el código para leer los libros **completos** con `f.read()`, se compara esa misma consulta contra novelas gigantescas de, por ejemplo, 100,000 palabras. Para evitar que los libros muy largos dominen a los cortos simplemente por tener más texto, el Modelo Vectorial aplica una técnica llamada **Normalización de la Magnitud** al calcular el Coseno:
 
$$sim(\vec{q},\vec{d})=\frac{\vec{q}\cdot\vec{d}}{||\vec{q}||\cdot||\vec{d}||}$$
 
El divisor $||\vec{d}||$ representa la "longitud" geométrica del libro. Cuando se lee un libro entero, esa magnitud se convierte en un número gigantesco. Al dividir por un valor tan grande, el puntaje de similitud final se **diluye** y se vuelve mucho más pequeño, pudiendo caer hasta valores como `0.09`. Esto es el comportamiento esperado y correcto del modelo: está comparando de forma justa sin importar el tamaño del documento.

## Ejecución Principal

El bloque final `__main__` utiliza la librería nativa `os` de Python para construir rutas de archivos dinámicas (`os.path.join`).

Esto actúa como un **GPS interno**: el código averigua automáticamente en qué carpeta y sistema operativo (Windows, Mac, Linux o contenedores Docker) se está ejecutando, garantizando que el proyecto funcione en cualquier computadora sin tener que modificar el código manualmente.
