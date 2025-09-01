.
├── .DS_Store
├── .github
│   └── workflows
│       ├── ci.yml
│       └── deploy-docks.yml
├── .gitignore
├── data
├── docker
│   ├── .dockerignore
│   ├── docker-compose.dev.yml
│   ├── docker-compose.prod.yml
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── Dockerfile.dev
│   ├── Dockerfile.jupyter
│   └── entrypoint.sh
├── docs
│   ├── docs
│   │   ├── api
│   │   │   ├── index.md
│   │   │   └── reference.md
│   │   ├── assets
│   │   │   ├── custom.css
│   │   │   └── logo.svg
│   │   ├── contributing.md
│   │   ├── examples.md
│   │   ├── index.md
│   │   └── quickstart.md
│   ├── mkdocs.yml
│   └── requirements.txt
├── models
├── notebooks
│   ├── 01_fundamentals
│   │   ├── 01_introduction_to_tensors.ipynb
│   │   ├── 02_gradient_computation.ipynb
│   │   ├── 03_custom_autograd_functions.ipynb
│   │   └── 04_backpropagation_visualization.ipynb
│   ├── 02_neural_networks
│   │   ├── 05_mlp_from_scratch.ipynb
│   │   ├── 06_advanced_architectures.ipynb
│   │   └── 07_training_techniques.ipynb
│   ├── 03_computer_vision
│   │   ├── 08_cnn_fundamentals.ipynb
│   │   ├── 09_modern_cnn_architectures.ipynb
│   │   └── 10_computer_vision_projects.ipynb
│   ├── 04_natural_language_processing
│   │   ├── 11_rnn_lstm_fundamentals.ipynb
│   │   ├── 12_sequence_to_sequence.ipynb
│   │   ├── 13_sentiment_analysis_project.ipynb
│   │   └── 14_transformer_from_scratch.ipynb
│   ├── 05_generative_models
│   │   ├── 15_gan_fundamentals.ipynb
│   │   └── 16_advanced_gans_vaes.ipynb
│   ├── 06_optimization_deployment
│   │   ├── 17_model_optimization.ipynb
│   │   ├── 18_model_serving_apis.ipynb
│   │   ├── 19_monitoring_mlops.ipynb
│   │   └── 20_cloud_deployment.ipynb
│   ├── 07_advanced_projects
│   │   ├── 21_image_classification_project.ipynb
│   │   ├── 22_text_generation_project.ipynb
│   │   └── 23_recommendation_system.ipynb
│   ├── 08_advanced_topics
│   │   ├── 24_advanced_techniques.ipynb
│   │   └── 25_research_applications.ipynb
│   └── capstone_projects
│       ├── 26_Capstone_part1_multimodal_system.ipynb
│       └── 27_Capstone_part2_production_mlops.ipynb
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
├── setup.py
├── src
│   ├── __init__.py
│   ├── advanced
│   │   ├── __init__.py
│   │   ├── deployment.py
│   │   ├── gan_utils.py
│   │   └── optimization.py
│   ├── computer_vision
│   │   ├── __init__.py
│   │   ├── augmentation.py
│   │   ├── datasets.py
│   │   ├── models.py
│   │   └── transforms.py
│   ├── fundamentals
│   │   ├── __init__.py
│   │   ├── autograd_helpers.py
│   │   ├── math_utils.py
│   │   └── tensor_ops.py
│   ├── neural_networks
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   ├── models.py
│   │   ├── optimizers.py
│   │   └── training.py
│   ├── nlp
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── models.py
│   │   ├── text_utils.py
│   │   └── tokenization.py
│   └── utils
│       ├── __init__.py
│       ├── data_utils.py
│       ├── io_utils.py
│       ├── metrics.py
│       └── visualization.py
└── tests
    ├── __init__.py
    ├── conftest.py
    ├── pytest.ini
    ├── run_tests.py
    ├── test_advanced
    │   └── test_gan_utils.py
    ├── test_computer_vision
    │   └── test_transforms.py
    ├── test_fundamentals
    │   └── test_tensor_ops.py
    ├── test_integration.py
    ├── test_neural_networks
    │   └── test_models.py
    ├── test_nlp
    │   └── test_tokenization.py
    └── test_utils
        ├── test_data_utils.py
        └── test_visualization.py

34 directories, 94 files
