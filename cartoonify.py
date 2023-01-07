import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import time


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


def read_image(path):
    img = cv2.imread(path)
    return img


def upload():
    image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
    if image_file is not None:
        file_details = {"FileName": image_file.name, "FileType": image_file.type}
        st.write(file_details)
        if st.button("Cartoonify!"):
            img = load_image(image_file)
            st.subheader("Original Image: ")
            st.image(img)
            with open(os.path.join(r"..\Cartoonify_Streamlit\Saved Image", image_file.name), "wb") as f:
                f.write(image_file.getbuffer())
                path = os.path.join(r"..\Cartoonify_Streamlit\Saved Image", image_file.name)
            # st.success("Saved File")
            return path


def quantization(img, k):
    iD = np.float32(img).reshape((-1, 3))
    iC = (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(iD, k, None, iC, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    iN = center[label.flatten()]
    iN = iN.reshape(img.shape)
    return iN


def img_edge(img, edge_width, blur):
    # convert color image to gray scale
    gC = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # st.image(gC)
    # covert gray scale image to blur image
    gB = cv2.medianBlur(gC, blur)
    # st.image(gB)
    # calculate and store the image edges
    iE = cv2.adaptiveThreshold(gB, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, edge_width, blur)
    return iE


def main():
    latest_iteration = st.empty()
    bar = st.progress(0)
    for i in range(100):
        latest_iteration.text(f'Loading... {i + 1}')
        bar.progress(i + 1)
        time.sleep(0.01)
    page = st.sidebar.selectbox(
        'Select a page?',
        ('Cartoonify', 'About this Project', 'About Me')
    )

    if page == "Cartoonify":
        st.title("Cartoonify Image...")
        st.write("-By Piyush Chauhan")

        path = upload()
        image = read_image(path)
        cluster = st.sidebar.slider('Numbers of clusters', 1, 40, 21)
        ew = st.sidebar.slider('Edge Width', 3, 9, 9, 2)
        bv = st.sidebar.slider('Blur Value', 1, 9, 9, 2)
        edge_width = ew
        blur_value = bv
        totalColors = cluster

        if image is not None:
            img_Edge = img_edge(image, edge_width, blur_value)
            # st.image(img_Edge)
            image = quantization(image, totalColors)
            # st.image(image)
            blurred = cv2.medianBlur(image, blur_value)
            # st.image(blurred)
            cartoonify = cv2.bitwise_and(blurred, blurred, mask=img_Edge)

            st.subheader("Cartoonified Image: ")
            st.image(cartoonify)
            cv2.imwrite('cartoonifystreamlit.jpg', cartoonify)
            if st.button("Save?"):
                st.success("Saved File")


    elif page == "About this Project":
        st.title('About this Project')
        st.markdown(
            "This application is for Converting Image into cartoon image project but converted image is not in "
            "the perfect color. Image convert will get automatically saved in the current working directory, "
            "and you have the option to directly upload the image and convert it. However the uploaded image will be "
            "saved in the Saved Image folder. ")


    elif page == "About Me":
        st.title('About Me')
        st.markdown("Hi There,")
        st.markdown(
            "My name is Piyush Chauhan,I am a student of Graphic era deemed University, Dehradun. I have made this "
            "beautiful project in Semester 5 as a Mini Project for the partial fulfillment of degree of btech in "
            "Computer Science. I have been working in field of Machine Learning for almost a year now. I Learn new "
            "things by trying it out.")
        st.markdown("We can connect over LinkedIn at : https://www.linkedin.com/in/piyush-chauhan-a17a7b217/")
        st.markdown("You can reach out to me at : pchauhan7289@gmail.com")


if __name__ == '__main__':
    main()
