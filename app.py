import pickle
import re
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Analyse de sentiment d’un avis", layout="centered")
st.html("""
<style>
    html {font-size: 90%;}
    [data-testid="stMainBlockContainer"] {
        max-width: 52.5rem;
        padding: .85rem 7.5rem 2rem 1rem;
    }
    @media (max-width: 52.5rem) {
        [data-testid="stMainBlockContainer"] {max-width: 46rem; padding-right: 1rem;}
    }
    [data-testid="stAppDeployButton"] {display: none;}
    [data-testid="stHeader"] {background: transparent; pointer-events: none;}
    [data-testid="stToolbar"] {pointer-events: auto;}
    .stButton button[kind="primary"] {
        background-color: #c0404d;
        border-color: #c0404d;
        color: white;
    }
    .stButton button[kind="primary"]:hover {background-color: #a93440;}
    [data-testid="stProgress"] [role="progressbar"] > div > div {background-color: #3282df;}
</style>
""")

st.title("Analyse de sentiment d’un avis")
st.markdown("Collez un avis **en anglais** ou importez un fichier texte pour obtenir sa catégorie.")


def nettoyer(txt):
    txt = txt.lower().replace("\\n", " ")
    txt = re.sub(r"[^a-z\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


dossier = Path(__file__).resolve().parent
fiche = dossier / "model_info.pkl"
if not fiche.is_file():
    st.error("Modèle introuvable. Exécute la partie 4 du notebook pour le sauvegarder.")
    st.stop()

with fiche.open("rb") as fichier:
    info = pickle.load(fichier)


@st.cache_resource
def charger(type_modele, version_sauvegarde):
    if type_modele == "rnn":
        from tensorflow.keras.models import load_model

        with (dossier / "tokenizer.pkl").open("rb") as fichier:
            tokenizer = pickle.load(fichier)
        return load_model(dossier / "rnn_yelp.keras"), tokenizer

    import joblib
    return joblib.load(dossier / "model_sklearn.joblib"), None


def analyser(avis):
    if info["type"] == "rnn":
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        modele, tokenizer = charger("rnn", fiche.stat().st_mtime_ns)
        sequence = pad_sequences(tokenizer.texts_to_sequences([avis]), maxlen=info["MAXLEN"],
                                 padding="post", truncating="post")
        return float(modele.predict(sequence, verbose=0)[0, 0]), "Probabilité associée", "Probabilités estimées par le LSTM"

    modele, _ = charger("sklearn", (dossier / "model_sklearn.joblib").stat().st_mtime_ns)
    if hasattr(modele, "predict_proba"):
        positif = float(modele.predict_proba([avis])[0, list(modele.classes_).index(1)])
        legende = ("Probabilités calibrées à partir des scores du SVM" if info["nom"] == "SVM"
                   else f"Probabilités estimées par le modèle {info['nom']}")
        return positif, "Probabilité associée", legende

    # Les anciens exports du TP contiennent un LinearSVC sans calibration.
    from scipy.special import expit

    score = float(expit(modele.decision_function([avis])[0]))
    return score, "Score associé", "Scores indicatifs de l’ancien SVM sans calibration"


texte = st.text_area("Votre texte", height=160)
fichier_texte = st.file_uploader("Ou importer un fichier texte", type=["txt"],
                                 help="Si vous fournissez les deux, le fichier est analysé en priorité.")

if st.button("Analyser", type="primary", width="stretch"):
    if fichier_texte is not None:
        donnees = fichier_texte.getvalue()
        encodage = "utf-16" if donnees.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        try:
            texte = donnees.decode(encodage)
        except UnicodeDecodeError:
            st.error("Le fichier est illisible. Enregistrez-le au format texte UTF-8 puis réessayez.")
            st.stop()

    avis = nettoyer(texte)
    if not avis:
        st.warning("Saisissez un avis en anglais ou importez un fichier .txt contenant du texte.")
        st.stop()

    positif, libelle, legende = analyser(avis)
    negatif = 1 - positif
    categorie = "Positif" if positif >= 0.5 else "Négatif"
    confiance = max(positif, negatif)

    st.subheader(f"Catégorie : {categorie}")
    st.metric(libelle, f"{confiance:.1%}")
    st.progress(confiance)
    st.caption(legende)
    st.json({"Négatif": f"{negatif:.1%}", "Positif": f"{positif:.1%}"})
