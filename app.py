import streamlit as st
import os, tempfile 
from src.multistd_classifier import MultiStudentClassifier
from src.video_classifier import VideoClassifier

st.set_page_config(
    page_title="Smart Classroom Attention Monitoring",
    layout="wide"
)

st.title("Smart Classroom Attention Monitoring System")

st.write(
    "Upload a classroom video and analyze student attention."
)

mode = st.selectbox(
    "Select Analysis Mode",
    ["Single Student", "Multi Student"]
)

uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        temp_path=tmp.name
    st.success("Video uploaded successfully!")
    st.video(temp_path)
    if st.button("Analyze Button"):
        st.write("Processing started.........")
        if mode=="Single Student":
            classifier=VideoClassifier()
            with st.spinner("Analyzing video ......."):
                result=classifier.classify_video(temp_path)
                st.write(f"Label: {result['final_label']}")
                st.write(f"Attentive Frames: {result['attentive_frames']}")
                st.write(f"Total Frames: {result['total_frames']}")

        else:
            classifier=MultiStudentClassifier()
            with st.spinner("Analyzing video ......."):
                result=classifier.classify_video(temp_path)
                st.write(f"Frames Processed: {result['frames_processed']}")
                st.write(f"Students Detected: {result['students_detected']}")
                for face_id,counts in result["per_student"].items():
                    label = "Attentive" if counts["attention_rate"] >= 70 else "Distracted"
                    st.header(face_id.replace('_', ' ').title())
                    st.caption(f"Status Label : {label}")
                    st.metric(label="Attention Rate", value=f"{counts['attention_rate']:.2f}%")
                    st.metric(label="Attentive Frames", value=counts['attentive_frames'])
                    st.metric(label="Distracted Frames", value=counts['distracted_frames'])



                    # st.markdown(f"""
                    #     ### {face_id.replace('_', ' ').title()}
                    #     **Label:** {label}
                    #     **Attention Rate:** {counts['attention_rate']:.2f}%
                    #     **Attentive Frames:** {counts['attentive_frames']}
                    #     **Distracted Frames:** {counts['distracted_frames']}
                    #     """)


        # with st.spinner("Analyzing video ......."):
        #     result=classifier.classify_video(temp_path)

        # st.success("Analysis complete!")
        # st.write(result)
       


# if uploaded_file:
#     temp="temp"
#     os.makedirs(temp,exist_ok=True)
#     temp_path=os.path.join(temp,uploaded_file.name)
#     with open(temp_path,"wb") as f:
#         f.write(uploaded_file.getbuffer())
#     st.success("Video uploaded successfully!")
#     st.video(temp_path)
#     if st.button("Analyze Button"):
#         st.write("Processing started.........")

#         if mode=="Single Student":
#             classifier=VideoClassifier()
#         else:
#             classifier=MultiStudentClassifier()
#         with st.spinner("Analyzing video ......."):
#             result=classifier.classify_video(temp)

#         st.success("Analysis complete!")
#         st.write(result)




    st.write("Selected Mode:", mode)

