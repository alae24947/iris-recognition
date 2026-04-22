# iris-recognition
# 🏦 Iris Bank Microservices Pipeline

A microservices-based banking system integrating authentication, banking operations, AI iris recognition, notifications, and a web UI using Docker-based infrastructure.

---

## 📁 Project Structure

```bash
iris_bank_Microservices_Pipeline/
│
├── README.md
├── REPORT.pdf
├── .gitignore
├── docker-compose.yml
├── docker-compose.prod.yml
│
├── services/
│   ├── auth-service/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   │   ├── login.py
│   │   │   │   └── register.py
│   │   │   ├── models/
│   │   │   │   └── user.py
│   │   │   ├── services/
│   │   │   │   └── jwt_handler.py
│   │   │   └── config.py
│   │   └── tests/
│   │
│   ├── banking-service/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── manage.py
│   │   ├── core/
│   │   │   ├── models.py
│   │   │   ├── views.py
│   │   │   ├── serializers.py
│   │   │   └── urls.py
│   │   ├── iris_bank_settings/
│   │   └── tests/
│   │
│   ├── iris-ml-service/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   │   ├── main.py
│   │   │   ├── model_loader.py
│   │   │   ├── inference.py
│   │   │   └── utils.py
│   │   ├── models/
│   │   │   └── best_siamese.pth
│   │   └── tests/
│   │
│   ├── notification-worker/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── src/
│   │   │   ├── worker.py
│   │   │   ├── tasks/
│   │   │   │   ├── send_email.py
│   │   │   │   └── log_transaction.py
│   │   │   └── config.py
│   │   └── tests/
│   │
│   └── web-ui/
│       ├── Dockerfile
│       ├── package.json
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   │   ├── Login.js
│       │   │   ├── Dashboard.js
│       │   │   └── Transfer.js
│       │   ├── services/
│       │   │   └── api.js
│       │   └── App.js
│       └── public/
│
├── infrastructure/
│   ├── consul/
│   │   ├── Dockerfile
│   │   ├── config/
│   │   │   └── consul.hcl
│   │   └── data/
│   │
│   ├── rabbitmq/
│   │   ├── Dockerfile
│   │   ├── config/
│   │   │   └── rabbitmq.conf
│   │   └── management_plugin/
│   │
│   └── traefik/
│       ├── Dockerfile
│       ├── dynamic/
│       │   └── file_config.yml
│       └── static/
│           └── traefik.yml
│
├── scripts/
│   ├── deploy/
│   │   ├── deploy_server_1.sh
│   │   ├── deploy_server_2.sh
│   │   ├── deploy_server_3.sh
│   │   └── init_cluster.sh
│   ├── build_all.sh
│   └── seed_data.py
│
└── docs/
    ├── architecture_diagram.png
    ├── api_swagger.json
    ├── deployment_guide.md
    └── user_manual.md
