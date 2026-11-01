/**
 * Service centralisé de communication avec le backend Python.
 *
 * Rôle : un seul endroit connaît l'URL de l'API et le format des requêtes.
 * Les composants React n'appellent jamais fetch() directement.
 */

// VITE_API_URL est défini dans .env (voir .env.example).
// Attention : tout ce qui commence par VITE_ est visible dans le navigateur.
// N'y mettez jamais de secret.
// Un "VITE_API_URL=/" (voir .env.production) signifie "même origine que la
// page" : c'est le mode utilisé quand FastAPI sert le build. On retire le
// slash final pour éviter les doubles slashs dans les URLs.
const API_URL = (import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000').replace(
  /\/$/,
  '',
)

/**
 * Demande une prédiction au backend.
 *
 * @param {{superficie_m2: number, nb_pieces: number, age_annees: number, quartier: string}} donnees
 * @returns {Promise<{prix_estime: number, unite: string, intervalle_min?: number, intervalle_max?: number}>}
 * @throws {Error} message lisible par l'utilisateur
 */
export async function predirePrix(donnees) {
  let reponse

  try {
    reponse = await fetch(`${API_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(donnees),
    })
  } catch {
    // Erreur réseau : backend éteint, mauvaise URL, CORS...
    throw new Error(
      "Impossible de joindre l'API. Vérifiez que le backend est démarré.",
    )
  }

  // FastAPI renvoie un JSON même sur erreur : on essaie de lire le détail.
  let corps = null
  try {
    corps = await reponse.json()
  } catch {
    corps = null
  }

  if (!reponse.ok) {
    if (reponse.status === 422) {
      throw new Error('Données invalides. Vérifiez les valeurs saisies.')
    }
    if (reponse.status === 503) {
      throw new Error("Le modèle n'est pas disponible sur le serveur.")
    }
    const detail =
      typeof corps?.detail === 'string' ? corps.detail : null
    throw new Error(detail ?? `Erreur serveur (code ${reponse.status}).`)
  }

  if (!corps?.prediction || typeof corps.prediction.prix_estime !== 'number') {
    throw new Error('Réponse inattendue du serveur.')
  }

  return corps.prediction
}

/**
 * Récupère la description du modèle servi (métriques, jeu de données).
 *
 * @returns {Promise<object|null>} null si l'information est indisponible.
 */
export async function obtenirInfosModele() {
  try {
    const reponse = await fetch(`${API_URL}/api/model-info`)
    if (!reponse.ok) return null
    return await reponse.json()
  } catch {
    // Optionnel : l'interface reste utilisable sans cet encart.
    return null
  }
}
