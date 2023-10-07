import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, MultiHeadAttention, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import textstat as ts
from pdfminer.high_level import extract_text
from io import StringIO
import docx2txt
import requests
from bs4 import BeautifulSoup as bs
import language_tool_python

# Define the training data
training_data = [
    ("I was walking down the street and a group of men approached me.", "biased"),
    ("The police officer pulled me over because of the color of my skin.", "biased"),
    ("My friend was denied a job because of their race.", "biased"),
    ("I don't see color, I treat everyone the same.", "unbiased"),
    ("We need to have more diversity in our workplace.", "unbiased"),
    ("I believe in affirmative action to address historical injustices.", "unbiased"),
    ("All lives matter, not just black lives.", "biased"),
    ("We need to stop using racial slurs, it's not okay.", "unbiased"),
    ("The media always portrays people of color as criminals.", "biased"),
    ("We need to have more open conversations about race.", "unbiased"),
    ("Racial profiling is a serious issue that needs to be addressed.", "biased"),
    ("Promoting equal opportunities for all is essential for a just society.", "unbiased"),
    ("Discrimination based on race is a violation of human rights.", "unbiased"),
    ("Stereotyping individuals based on their race perpetuates inequality.", "unbiased"),
    ("Systemic racism has deep-rooted impacts on marginalized communities.", "biased"),
    ("Acknowledging privilege is the first step towards creating an equitable society.", "unbiased"),
    ("Prejudice and bias have no place in our society.", "unbiased"),
    ("Institutionalized discrimination affects access to opportunities for minority groups.", "biased"),
    ("Promoting diversity and inclusion leads to innovation and growth.", "unbiased"),
    ("Racial profiling leads to unjust treatment and undermines trust in law enforcement.", "biased"),
    ("The American Revolution was a colonial revolt that took place between 1765 and 1783. The American Patriots in the Thirteen Colonies won independence from Great Britain, becoming the United States of America.", "historical"),
    ("The French Revolution was a period of social and political upheaval in France from 1789 to 1799 that profoundly affected French and world history. It led to the end of the monarchy and to many wars.", "historical"),
    ("The Scientific Revolution was a series of events that marked the emergence of modern science during the early modern period, when developments in mathematics, physics, astronomy, biology (including human anatomy) and chemistry transformed the views of society about nature.", "historical"),
    ("The Industrial Revolution was the transition to new manufacturing processes in Europe and the United States, in the period from between 1760 to 1820 and 1840. This transition included going from hand production methods to machines, new chemical manufacturing and iron production processes, improved efficiency of water power, the increasing use of steam power, the development of machine tools and the rise of the mechanized factory system.", "historical"),
    ("The Roman Empire was the post-Republican period of ancient Rome. It is characterized by the rule of emperors who were autocratic and dynastic. The Roman Empire lasted from 27 BC to 476 AD.", "historical"),
    ("The Ming dynasty was the ruling dynasty of China from 1368 to 1644, following the collapse of the Mongol-led Yuan dynasty. The Ming dynasty was the last imperial dynasty in China ruled by ethnic Han Chinese.", "historical"),
    ("The Renaissance was a period in European history from the 14th to the 17th century, regarded as the cultural bridge between the Middle Ages and modern history. It started in Italy and spread across Europe, bringing renewed interest in classical art, literature, and learning.", "historical"),
    ("The Cold War was a geopolitical conflict between the Soviet Union and the United States, lasting from the end of World War II until the dissolution of the Soviet Union in 1991. It was characterized by a nuclear arms race, proxy wars, and ideological competition.", "historical"),
    ("The Romans conquered Britain in 43 AD.", "historical"),
    ("John F. Kennedy was the 35th President of the United States.", "historical"),
    ("The Great Wall of China was built in the 7th century BC.", "historical"),
    ("The Beatles were a British rock band formed in Liverpool in 1960.", "historical"),
    ("I was walking down the street and a group of men approached me.", "non-historical"),
    ("The police officer pulled me over because of the color of my skin.", "non-historical"),
    ("My friend was denied a job because of their race.", "non-historical"),
    ("I don't see color, I treat everyone the same.", "non-historical"),
    ("We need to have more diversity in our workplace.", "non-historical"),
    ("I believe in affirmative action to address historical injustices.", "non-historical"),
    ("All lives matter, not just black lives.", "non-historical"),
    ("We need to stop using racial slurs, it's not okay.", "non-historical"),
    ("The media always portrays people of color as criminals.", "non-historical"),
    ("We need to have more open conversations about race.", "non-historical"),
    ("People with disabilities", "disability"),
    ("Disclosing a disability", "disability"),
    ("Disorder", "disability"),
    ("Epileptic", "disability"),
    ("Handicapped parking", "disability"),
    ("Help (this suggests a weakness)", "disability"),
    ("Hidden impairment", "disability"),
    ("Lame", "disability"),
    ("Mentally retarded, retard, slow", "disability"),
    ("Midget", "disability"),
    ("Normal, able-bodied", "unbiased"),
    ("Sighted person, or hearing person, or neurotypical person", "unbiased"),
    ("Schizophrenic", "disability"),
    ("Spastic", "disability"),
    ("Special needs", "disability"),
    ("Symptoms [of a condition]", "disability"),
    ("Visual impairment, people who are visually impaired (this can be interpreted as indicating someone who looks visually disfigured or diminished in some way)", "disability"),
    ("Wheelchair bound, confined to a wheelchair", "disability"),
    (" not he, she", "gender-inclusion"),
    ("not husband, wife","gender-inclusion"),
    ("not mother, father","gender-inclusion"),
    ("not opposite sex","gender-inclusion"),
    ("not businessman, businesswoman","gender-inclusion"),
    (" not policeman, policewoman","gender-inclusion"),
    ("not chairman, chairwoman","gender-inclusion"),
    ("not man-made","gender-inclusion"),
    ("equality, diversity, equity", "gender-inclusion"),
    ("women, men, non-binary, genderqueer, transgender, LGBTQ", "gender-inclusion"),
]


# Tokenize the training data
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters="", oov_token="<OOV>")
tokenizer.fit_on_texts([data[0] for data in training_data])

# Convert the training data to sequences of tokens
training_sequences = [tokenizer.texts_to_sequences([data[0]])[0] for data in training_data]

# Pad the sequences to a fixed length
max_length = max([len(sequence) for sequence in training_sequences])
training_sequences = tf.keras.preprocessing.sequence.pad_sequences(training_sequences, maxlen=max_length)

# Split the data into input and output sequences
training_inputs = training_sequences[:, :-1]
#training_outputs = tf.keras.utils.to_categorical([1 if data[1] == "biased" else 2 if data[1] == "unbiased" else 3 if data[1] == "historical" else 4 for data in training_data], num_classes=5)

# Calculate the number of classes dynamically
num_classes = len(set(data[1] for data in training_data))

# Convert class labels to one-hot encoded vectors
training_outputs = tf.keras.utils.to_categorical(
    [i for i in range(num_classes)], num_classes=num_classes
)



# Define the model parameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
num_heads = 4
dense_dim = 256
dropout_rate = 0.1

# Define the input layer
inputs = Input(shape=(None,), dtype="int32")

# Define the embedding layer
embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

# Define the transformer layer
multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
dense_1 = Dense(units=dense_dim, activation="relu")
dropout_1 = Dropout(rate=dropout_rate)
normalize_1 = LayerNormalization()
dense_2 = Dense(units=num_classes, activation="softmax")
dropout_2 = Dropout(rate=dropout_rate)
normalize_2 = LayerNormalization()

# Define the forward pass through the transformer layer
x = multi_head_attention(embedding, embedding)
x = normalize_1(x + embedding)
x = dense_1(x)
x = dropout_1(x)
x = normalize_2(x + x)
x = dense_2(x)
outputs = dropout_2(x)

# Define the full model
model = Model(inputs=inputs, outputs=outputs[:, -1, :])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
x_train = training_inputs[:7]
y_train = training_outputs[:7]
x_val = training_inputs[7:9]
y_val = training_outputs[7:9]
#model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Evaluate the model on the test set
x_test = training_inputs[9:]
y_test = training_outputs[9:]
#loss, accuracy = model.evaluate(x_test, y_test)
#st.write(f'Test loss: {loss:.2f}, Test accuracy: {accuracy:.2f}')

# Create the Streamlit app
st.title("YZEN DEI Project")

# Define the layout selection
layout_option = st.sidebar.radio("Select Layout", ["Text Input", "Upload and Analyze File"])

if layout_option == "Text Input":
    # Create a text input for user input
    user_input = st.text_area("Enter a text", height=300)

    # Preprocess the user input
    user_sequences = tokenizer.texts_to_sequences([user_input])
    user_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(user_sequences, maxlen=max_length)

    # Make a prediction on the user input
    prediction = model.predict(user_padded_sequences)
    percentage_biased = prediction[0][0] * 100
    percentage_unbiased = prediction[0][1] * 100
    percentage_historical = prediction[0][2] * 100
    percentage_non_historical = prediction[0][3] * 100
    percentage_disability = prediction[0][4] * 100  # New line for disability
    percentage_gender_inclusion = prediction[0][5] * 100 


    # Determine the predicted label
    if percentage_biased > percentage_unbiased and percentage_biased > percentage_historical and percentage_biased > percentage_non_historical:
        label = "biased"
    elif percentage_disability > percentage_biased and percentage_disability > percentage_unbiased and percentage_disability > percentage_historical and percentage_disability > percentage_non_historical:
        label = "disability"
    elif percentage_gender_inclusion > percentage_disability and percentage_gender_inclusion > percentage_unbiased and percentage_gender_inclusion > percentage_historical and percentage_gender_inclusion > percentage_non_historical:
        label = "gender-inclusion"
    elif percentage_unbiased > percentage_biased and percentage_unbiased > percentage_historical and percentage_unbiased > percentage_non_historical:
        label = "unbiased"
    elif percentage_historical > percentage_biased and percentage_historical > percentage_unbiased and percentage_historical > percentage_non_historical:
        label = "historical"
    else:
        label = "non-historical"

    # Display the predicted label and percentages in the sidebar
    st.sidebar.subheader("Prediction")
    st.sidebar.write(f"Biased: {percentage_biased:.2f}%")
    st.sidebar.write(f"Unbiased: {percentage_unbiased:.2f}%")
    st.sidebar.write(f"Disability: {percentage_disability:.2f}%")
    st.sidebar.write(f"Gender Inclusive Language: {percentage_gender_inclusion:.2f}%")
    st.sidebar.write(f"Historical: {percentage_historical:.2f}%")
    st.sidebar.write(f"Non-historical: {percentage_non_historical:.2f}%")

# elif layout_option == "Upload and Analyze File":
#     # Create file upload widget
#     uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

#     if uploaded_file is not None:
#         # Process the uploaded file
#         if uploaded_file.type == "text/plain":
#             # Text file
#             text_content = uploaded_file.read()
#             st.subheader("Uploaded Text Content")
#             st.write(text_content)
#             st.write('Text Statistics')
            
#         elif uploaded_file.type == "application/pdf":
#             # PDF file
#             pdf_text = extract_text(uploaded_file)
#             st.subheader("Extracted Text from PDF")
#             st.write(pdf_text)
#             st.write('Text Statistics')
            
#         elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
#             # DOCX file
#             docx_text = docx2txt.process(uploaded_file)
#             st.subheader("Extracted Text from DOCX")
#             st.write(docx_text)
#             st.write('Text Statistics')
            
#         else:
#             st.warning("Unsupported file format")

elif layout_option == "Upload and Analyze File":
    # Create file upload widget
    uploaded_file = st.file_uploader("Upload a file", type=["txt", "pdf", "docx"])

    if uploaded_file is not None:
        content = None  # Initialize the content variable

        if uploaded_file.type == "text/plain":
            # Text file
            content = uploaded_file.read()
            st.subheader("Uploaded Text Content")
            st.write(content)
            
        elif uploaded_file.type == "application/pdf":
            # PDF file
            content = extract_text(uploaded_file)
            st.subheader("Extracted Text from PDF")
            st.write(content)
            
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            # DOCX file
            content = docx2txt.process(uploaded_file)
            st.subheader("Extracted Text from DOCX")
            st.write(content)
        
        else:
            st.warning("Unsupported file format")

        if content is not None:
            # Make a prediction on the uploaded text content
            uploaded_sequences = tokenizer.texts_to_sequences([content])
            uploaded_padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(uploaded_sequences, maxlen=max_length)
            prediction = model.predict(uploaded_padded_sequences)
            percentage_biased = prediction[0][0] * 100
            percentage_unbiased = prediction[0][1] * 100
            percentage_historical = prediction[0][2] * 100
            percentage_non_historical = prediction[0][3] * 100
            percentage_disability = prediction[0][4] * 100  # New line for disability
            percentage_gender_inclusion = prediction[0][5] * 100 


            # Determine the predicted label
            if percentage_biased > percentage_unbiased and percentage_biased > percentage_historical and percentage_biased > percentage_non_historical:
                label = "biased"
            elif percentage_disability > percentage_biased and percentage_disability > percentage_unbiased and percentage_disability > percentage_historical and percentage_disability > percentage_non_historical:
                label = "disability"
            elif percentage_gender_inclusion > percentage_disability and percentage_gender_inclusion > percentage_unbiased and percentage_gender_inclusion > percentage_historical and percentage_gender_inclusion > percentage_non_historical:
                label = "gender-inclusion"
            elif percentage_unbiased > percentage_biased and percentage_unbiased > percentage_historical and percentage_unbiased > percentage_non_historical:
                label = "unbiased"
            elif percentage_historical > percentage_biased and percentage_historical > percentage_unbiased and percentage_historical > percentage_non_historical:
                label = "historical"
            else:
                label = "non-historical"

            # Display the predicted label and percentages for uploaded text content
            st.write("Prediction for Uploaded Text Content:")
            st.sidebar.subheader("Prediction")
            st.sidebar.write(f"Biased: {percentage_biased:.2f}%")
            st.sidebar.write(f"Unbiased: {percentage_unbiased:.2f}%")
            st.sidebar.write(f"Disability: {percentage_disability:.2f}%")
            st.sidebar.write(f"Gender Inclusive Language: {percentage_gender_inclusion:.2f}%")
            st.sidebar.write(f"Historical: {percentage_historical:.2f}%")
            st.sidebar.write(f"Non-historical: {percentage_non_historical:.2f}%")
