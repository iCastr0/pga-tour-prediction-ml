# 1. Cargar librerías necesarias
library(readr)
library(tidyverse)
library(caret)
library(randomForest)

# 2. Cargar el modelo entrenado
rf_model_loaded <- readRDS("modelo_rf_final.rds")

# 3. Cargar el dummy model (si no lo tenés guardado, recrealo)
selected_vars <- pga_data %>%
  select(Winner, sg_putt, sg_ott, sg_t2g, sg_total, strokes, course, purse, made_cut, n_rounds, season)

selected_vars_clean <- selected_vars %>% filter(!is.na(Winner)) %>% drop_na()

nuevo_jugador <- tibble(
  sg_putt = 0.85,
  sg_ott = 0.76,
  sg_t2g = 3.12,
  sg_total = 3.87,
  strokes = 282,
  course = factor("Muirfield Village Golf Club - Dublin, OH", levels = levels(selected_vars_clean$course)),
  purse = 12.0,
  made_cut = TRUE,
  n_rounds = 4,
  season = factor("2022", levels = levels(selected_vars_clean$season))
)

dummy_model <- dummyVars(~ ., data = selected_vars_clean %>% select(-Winner))

# Codificar nuevo jugador
nuevo_jugador_encoded <- predict(dummy_model, newdata = nuevo_jugador) %>% as.data.frame()

# Predicción
prob_win <- predict(rf_model_loaded, newdata = nuevo_jugador_encoded, type = "prob")[, "Yes"]
pred_clase <- predict(rf_model_loaded, newdata = nuevo_jugador_encoded)

cat("Probabilidad de que gane:", round(prob_win, 4), "\n","Predicción de clase:", as.character(pred_clase))
cat("Predicción de clase:", as.character(pred_clase), "\n")
 
