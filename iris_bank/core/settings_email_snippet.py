# ─────────────────────────────────────────────────────────────
#  CONFIGURATION EMAIL — à ajouter dans votre settings.py
# ─────────────────────────────────────────────────────────────

# ── Option 1 : Gmail (recommandé pour les tests) ─────────────
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'votre.email@gmail.com'        # ← votre adresse Gmail
EMAIL_HOST_PASSWORD = 'xxxx xxxx xxxx xxxx'      # ← mot de passe d'application Gmail
                                                 #   (pas votre vrai mdp)
DEFAULT_FROM_EMAIL = 'Algerie Bank <votre.email@gmail.com>'

# ── Comment générer un mot de passe d'application Gmail ──────
# 1. Activez la validation en 2 étapes sur votre compte Google
# 2. Allez sur : https://myaccount.google.com/apppasswords
# 3. Sélectionnez "Autre (nom personnalisé)" → "Algerie Bank"
# 4. Copiez le mot de passe à 16 caractères généré

# ── Option 2 : Console (affiche l'email dans le terminal) ────
# Utile pour le développement sans vrai serveur SMTP
# EMAIL_BACKEND = 'django.core.mail.backends.console.EmailBackend'
# DEFAULT_FROM_EMAIL = 'noreply@algeriebank.dz'

# ── Option 3 : Fichier (sauvegarde les emails dans /tmp) ─────
# EMAIL_BACKEND = 'django.core.mail.backends.filebased.EmailBackend'
# EMAIL_FILE_PATH = '/tmp/django-emails'
