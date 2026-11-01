import { useEffect, useState } from 'react'
import { obtenirInfosModele, predirePrix } from './services/predictionService'
import './App.css'

// Valeurs initiales du formulaire, au format texte (input contrôlé).
const formulaireInitial = {
  superficie_m2: '50',
  nb_pieces: '3',
  age_annees: '5',
  quartier: 'Cocody',
}

// Repli uniquement si /api/model-info est injoignable : la liste officielle
// vient des métadonnées du modèle, pour rester synchronisée avec l'entraînement.
const QUARTIERS_SECOURS = ['Plateau', 'Cocody', 'Marcory', 'Yopougon', 'Bingerville']

const fcfa = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 0 })

/**
 * Validation côté interface. Ce n'est PAS la sécurité : le backend revalide
 * tout avec Pydantic. Ici, on évite juste un aller-retour réseau inutile et on
 * affiche un message immédiat.
 *
 * @returns {string|null} message d'erreur, ou null si tout est correct.
 */
function valider(formulaire) {
  const superficie = Number(formulaire.superficie_m2)
  const pieces = Number(formulaire.nb_pieces)
  const age = Number(formulaire.age_annees)

  if (!formulaire.superficie_m2.trim() || !Number.isFinite(superficie)) {
    return 'Indiquez une superficie en m².'
  }
  if (superficie <= 0) return 'La superficie doit être supérieure à 0 m².'
  if (superficie > 2000) {
    return 'La superficie dépasse la plage couverte par le modèle (2 000 m²).'
  }
  if (!Number.isInteger(pieces) || pieces < 1 || pieces > 30) {
    return 'Le nombre de pièces doit être un entier entre 1 et 30.'
  }
  if (!Number.isFinite(age) || age < 0 || age > 200) {
    return "L'âge du bien doit être compris entre 0 et 200 ans."
  }
  return null
}

function App() {
  const [formulaire, setFormulaire] = useState(formulaireInitial)
  const [prediction, setPrediction] = useState(null)
  const [chargement, setChargement] = useState(false)
  const [erreur, setErreur] = useState(null)
  const [infosModele, setInfosModele] = useState(null)

  // Informations du modèle : chargées une fois au montage. Si l'API est
  // éteinte, l'encart est simplement absent — le formulaire reste utilisable.
  useEffect(() => {
    let actif = true
    obtenirInfosModele().then((infos) => {
      if (actif && infos) setInfosModele(infos)
    })
    return () => {
      actif = false
    }
  }, [])

  const quartiers =
    infosModele?.quartiers?.length > 0 ? infosModele.quartiers : QUARTIERS_SECOURS

  function gererChangement(evenement) {
    const { name, value } = evenement.target
    setFormulaire((precedent) => ({ ...precedent, [name]: value }))
  }

  async function gererSoumission(evenement) {
    evenement.preventDefault()
    if (chargement) return // empêche les doubles soumissions

    const messageValidation = valider(formulaire)
    if (messageValidation) {
      setPrediction(null)
      setErreur(messageValidation)
      return
    }

    setErreur(null)
    setPrediction(null)
    setChargement(true)

    // Les inputs renvoient du texte : on convertit vers les types attendus
    // par le schéma Pydantic du backend.
    const donnees = {
      superficie_m2: Number(formulaire.superficie_m2),
      nb_pieces: Number(formulaire.nb_pieces),
      age_annees: Number(formulaire.age_annees),
      quartier: formulaire.quartier,
    }

    try {
      const resultat = await predirePrix(donnees)
      setPrediction({ ...resultat, entrees: donnees }) // re-render
    } catch (e) {
      setErreur(e.message)
    } finally {
      setChargement(false)
    }
  }

  const aUnIntervalle =
    prediction &&
    typeof prediction.intervalle_min === 'number' &&
    typeof prediction.intervalle_max === 'number'

  // Position du prix dans sa fourchette, en % : sert au marqueur du graphique.
  const positionPrix = aUnIntervalle
    ? Math.min(
        100,
        Math.max(
          0,
          ((prediction.prix_estime - prediction.intervalle_min) /
            (prediction.intervalle_max - prediction.intervalle_min)) *
            100,
        ),
      )
    : 50

  const metriques = infosModele?.metriques_test ?? {}
  const candidats = Object.entries(infosModele?.selection_modele ?? {})

  return (
    <div className="page">
      <header className="entete">
        <p className="surtitre">Modèle scikit-learn · API FastAPI</p>
        <h1>Estimer le prix d&apos;un logement</h1>
        <p className="entete-texte">
          Renseignez les caractéristiques du bien. Le calcul est fait par un
          modèle entraîné en Python, interrogé via une API REST ; React se
          contente d&apos;afficher la réponse.
        </p>
      </header>

      <div className="grille">
        <form className="panneau formulaire" onSubmit={gererSoumission} noValidate>
          <h2 className="panneau-titre">Caractéristiques du bien</h2>

          <div className="champs">
            <div className="champ champ--large">
              <label htmlFor="superficie_m2">Superficie</label>
              <div className="champ-saisie">
                <input
                  id="superficie_m2"
                  name="superficie_m2"
                  type="number"
                  inputMode="decimal"
                  min="1"
                  max="2000"
                  step="0.1"
                  required
                  disabled={chargement}
                  value={formulaire.superficie_m2}
                  onChange={gererChangement}
                />
                <span className="unite">m²</span>
              </div>
            </div>

            <div className="champ">
              <label htmlFor="nb_pieces">Nombre de pièces</label>
              <div className="champ-saisie">
                <input
                  id="nb_pieces"
                  name="nb_pieces"
                  type="number"
                  inputMode="numeric"
                  min="1"
                  max="30"
                  step="1"
                  required
                  disabled={chargement}
                  value={formulaire.nb_pieces}
                  onChange={gererChangement}
                />
              </div>
            </div>

            <div className="champ">
              <label htmlFor="age_annees">Âge du bien</label>
              <div className="champ-saisie">
                <input
                  id="age_annees"
                  name="age_annees"
                  type="number"
                  inputMode="numeric"
                  min="0"
                  max="200"
                  step="1"
                  required
                  disabled={chargement}
                  value={formulaire.age_annees}
                  onChange={gererChangement}
                />
                <span className="unite">ans</span>
              </div>
            </div>

            <div className="champ champ--large">
              <label htmlFor="quartier">Quartier</label>
              <div className="champ-saisie">
                <select
                  id="quartier"
                  name="quartier"
                  required
                  disabled={chargement}
                  value={formulaire.quartier}
                  onChange={gererChangement}
                >
                  {quartiers.map((quartier) => (
                    <option key={quartier} value={quartier}>
                      {quartier}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          <button type="submit" className="bouton" disabled={chargement}>
            {chargement && <span className="rotateur" aria-hidden="true" />}
            {chargement ? 'Estimation en cours…' : 'Estimer le prix'}
          </button>

          {erreur && (
            <p className="alerte" role="alert">
              {erreur}
            </p>
          )}
        </form>

        <section className="panneau resultat" aria-live="polite">
          {!prediction && !chargement && (
            <div className="vide">
              <p className="vide-titre">Aucune estimation</p>
              <p>
                Renseignez la superficie, le nombre de pièces, l&apos;âge et le
                quartier, puis lancez le calcul.
              </p>
            </div>
          )}

          {chargement && (
            <div className="vide">
              <p className="vide-titre">Calcul en cours…</p>
              <p>Envoi des données à l&apos;API et prédiction du modèle.</p>
            </div>
          )}

          {prediction && (
            <>
              <p className="surtitre">Prix estimé</p>
              <p className="prix">
                {fcfa.format(prediction.prix_estime)}
                <span className="prix-unite">{prediction.unite}</span>
              </p>

              {aUnIntervalle && (
                <div className="fourchette">
                  <div className="fourchette-barre" aria-hidden="true">
                    <span
                      className="fourchette-marqueur"
                      style={{ left: `${positionPrix}%` }}
                    />
                  </div>
                  <div className="fourchette-bornes">
                    <span>{fcfa.format(prediction.intervalle_min)}</span>
                    <span className="fourchette-legende">intervalle 80 %</span>
                    <span>{fcfa.format(prediction.intervalle_max)}</span>
                  </div>
                </div>
              )}

              <ul className="recap">
                <li>
                  <span>Superficie</span>
                  <strong>{prediction.entrees.superficie_m2} m²</strong>
                </li>
                <li>
                  <span>Pièces</span>
                  <strong>{prediction.entrees.nb_pieces}</strong>
                </li>
                <li>
                  <span>Âge</span>
                  <strong>{prediction.entrees.age_annees} ans</strong>
                </li>
                <li>
                  <span>Quartier</span>
                  <strong>{prediction.entrees.quartier}</strong>
                </li>
              </ul>

              <p className="note">
                Estimation statistique du modèle, pas une valeur de marché
                garantie.
              </p>
            </>
          )}
        </section>
      </div>

      {infosModele && (
        <aside className="panneau modele">
          <h2 className="panneau-titre">Ce que vaut ce modèle</h2>

          <dl className="metriques">
            <div>
              <dt>Algorithme</dt>
              <dd>{infosModele.algorithme}</dd>
            </div>
            {typeof metriques.r2 === 'number' && (
              <div>
                <dt>R² (jeu de test)</dt>
                <dd>{metriques.r2.toFixed(3)}</dd>
              </div>
            )}
            {typeof metriques.mae === 'number' && (
              <div>
                <dt>Erreur moyenne</dt>
                <dd>
                  {fcfa.format(metriques.mae)} {infosModele.unite}
                </dd>
              </div>
            )}
          </dl>

          {candidats.length > 0 && (
            <div className="comparaison">
              <p className="comparaison-titre">
                Modèles comparés en validation croisée (R²)
              </p>
              <ul>
                {candidats
                  .sort((a, b) => b[1] - a[1])
                  .map(([nom, score]) => (
                    <li key={nom}>
                      <span>{nom}</span>
                      <span className="barre">
                        <span
                          className="barre-remplissage"
                          style={{ width: `${Math.max(0, score) * 100}%` }}
                        />
                      </span>
                      <strong>{score.toFixed(3)}</strong>
                    </li>
                  ))}
              </ul>
            </div>
          )}

          <p className="avertissement">
            Jeu de données : {infosModele.jeu_de_donnees}. Les chiffres
            ci-dessus valident la chaîne technique, pas la fiabilité
            commerciale des prix.
          </p>
        </aside>
      )}
    </div>
  )
}

export default App
