"use client";
import { useRef, useState } from "react";

export default function Home() {
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [loading, setLoading] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const startRecording = async () => {
    setPrediction("");
    setAudioUrl(null);
    setAudioBlob(null);
    audioChunksRef.current = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new window.MediaRecorder(stream);
    mediaRecorderRef.current.ondataavailable = (e) => {
      audioChunksRef.current.push(e.data);
    };
    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(audioChunksRef.current, { type: "audio/wav" });
      setAudioBlob(blob);
      setAudioUrl(URL.createObjectURL(blob));
      stream.getTracks().forEach((track) => track.stop());
    };
    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const sendAudio = async () => {
    if (!audioBlob) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.wav");
    const res = await fetch("https://YOUR-BACKEND-URL/predict", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setPrediction(data.prediction || data.error || "No prediction");
    setLoading(false);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioUrl(URL.createObjectURL(file));
      setAudioBlob(file);
      setPrediction("");
    }
  };

  return (
    <div className="max-w-xl mx-auto p-8">
      <h1 className="text-3xl font-bold text-blue-600 mb-4">🎙️ Speech Intent Recognition</h1>
      <div className="mb-4">
        <button
          onClick={recording ? stopRecording : startRecording}
          className="bg-blue-500 text-white px-4 py-2 rounded mr-2"
        >
          {recording ? "Stop Recording" : "Start Recording"}
        </button>
        <input type="file" accept="audio/*" onChange={handleFileChange} className="mt-2" />
      </div>
      {audioUrl && (
        <div className="mb-4">
          <audio src={audioUrl} controls />
          <button
            onClick={sendAudio}
            className="bg-green-500 text-white px-4 py-2 rounded ml-2"
            disabled={loading}
          >
            {loading ? "Processing..." : "Send for Prediction"}
          </button>
        </div>
      )}
      {prediction && (
        <div className="mt-4 p-4 bg-green-100 rounded">
          <strong>Prediction:</strong> {prediction}
        </div>
      )}
    </div>
  );
}