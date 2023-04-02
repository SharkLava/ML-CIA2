from flask import Flask, render_template, request, redirect, url_for, session
import os
import tensorflow as tf
import pickle
import numpy as np
import cv2
import mariadb

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the saved model from the pickle file
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the emotion labels
emotion_dict = {
    0: "angry",
    1: "crying",
    2: "embarassed",
    3: "happy",
    4: "pleased",
    5: "sad",
    6: "shock",
}


@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("upload"))
    else:
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        # Check the username and password against the database
        try:
            conn = mariadb.connect(
                user="vishal@localhost",
                password="",
                host="localhost",
                port=3306,
                database="users",
            )
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (request.form["username"], request.form["password"]),
            )
            user = cursor.fetchone()
            conn.close()
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB: {e}")
            return render_template(
                "login.html", error="Error connecting to the database"
            )

        if user is not None:
            # If the username and password are correct, save the username to the session
            session["username"] = user[1]
            return redirect(url_for("upload"))
        else:
            # If the username or password are incorrect, display an error message
            return render_template("login.html", error="Invalid username or password")
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        # Check if a file was uploaded
        if "image" not in request.files:
            return render_template("upload.html", error="No file selected")
        file = request.files["image"]
        if file.filename == "":
            return render_template("upload.html", error="No file selected")

        # Save the uploaded file to the "static" folder
        image_path = os.path.join(app.static_folder, file.filename)
        file.save(image_path)

        # Load the image to be predicted
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image to the same size as the images used to train the model
        img = cv2.resize(img, (128, 128))

        # Normalize the pixel values to be between 0 and 1
        img = img / 255.0

        # Add a batch dimension to the image
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=-1)

        # Use the model to make a prediction
        class_probs = model.predict(img)[0]

        # Get the index of the predicted class
        predicted_class = np.argmax(class_probs)

        # Get the predicted emotion and its probability
        emotion = emotion_dict[predicted_class]
        probability = class_probs[predicted_class]

        # Pass the prediction result to the template
        return render_template(
            "upload.html",
            image_filename=file.filename,
            emotion=emotion,
            probability=probability,
        )
    else:
        return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True)
