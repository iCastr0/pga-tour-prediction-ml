# =========================
#     MODELO PGA TOUR
# =========================

# 1. CARGA DE LIBRERÍAS ----
library(readr)
library(tidyverse)
library(janitor)
library(lubridate)
library(DataExplorer)
library(skimr)
library(caret)
library(glmnet)
library(randomForest)
library(pROC)
library(corrplot)

# 2. CARGA Y LIMPIEZA DEL DATASET ----
pga_data <- read_csv("ASA All PGA Raw Data - Tourn Level.csv") %>%
  clean_names() %>%
  mutate(
    Winner = ifelse(pos == 1, 1, 0),
    player_initial_last = as.factor(player_initial_last),
    player_id = as.factor(player_id),
    tournament_id = as.factor(tournament_id),
    tournament_name = as.factor(tournament_name),
    course = as.factor(course),
    date = as.Date(date),
    season = as.factor(season),
    made_cut = as.logical(made_cut),
    pos = as.numeric(pos),
    finish = as.factor(finish)
  ) %>%
  select(-starts_with("unnamed"))

# Guardar dataset limpio
write_csv(pga_data, "pga_tour_clean.csv")

# 3. EXPLORACIÓN INICIAL ----
glimpse(pga_data)
summary(pga_data)
skim(pga_data)

# Valores faltantes
pga_data %>%
  summarise_all(~sum(is.na(.))) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "missing_count") %>%
  arrange(desc(missing_count)) %>%
  print()

# Correlaciones
cor_matrix <- cor(pga_data %>% select(where(is.numeric)), use = "pairwise.complete.obs")
corrplot(cor_matrix, method = "color", type = "lower", tl.cex = 0.8)

# Visualización: SG total vs Posición
ggplot(pga_data, aes(x = sg_total, y = pos)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess") +
  labs(title = "Posición final vs. SG Total", x = "SG Total", y = "Posición Final") +
  theme_minimal()

# 4. SELECCIÓN Y PREPROCESAMIENTO DE VARIABLES ----
selected_vars <- pga_data %>%
  select(Winner, sg_putt, sg_ott, sg_t2g, sg_total, strokes, course, purse, made_cut, n_rounds, season) %>%
  filter(!is.na(Winner)) %>%
  drop_na()

# One-hot encoding
dummy_model <- dummyVars("~ .", data = selected_vars)
selected_vars_encoded <- predict(dummy_model, newdata = selected_vars) %>% as.data.frame()

# Separar train/test
set.seed(123)
train_index <- createDataPartition(selected_vars_encoded$Winner, p = 0.8, list = FALSE)
train_data <- selected_vars_encoded[train_index, ]
test_data <- selected_vars_encoded[-train_index, ]

# 5. MODELO: REGRESIÓN LOGÍSTICA ----
log_model <- glm(Winner ~ ., data = train_data, family = "binomial")
pred_probs <- predict(log_model, newdata = test_data, type = "response")
pred_classes <- ifelse(pred_probs > 0.5, 1, 0)

conf_matrix_log <- table(Predicted = pred_classes, Actual = test_data$Winner)
accuracy_log <- sum(diag(conf_matrix_log)) / sum(conf_matrix_log)
roc_log <- roc(test_data$Winner, pred_probs)
auc_log <- auc(roc_log)

# 6. MODELO: RANDOM FOREST ----
train_matrix <- model.matrix(Winner ~ . - 1, data = train_data)
test_matrix <- model.matrix(Winner ~ . - 1, data = test_data)

rf_model <- randomForest(
  x = train_matrix, 
  y = as.factor(train_data$Winner),
  ntree = 100, importance = TRUE
)

rf_preds <- predict(rf_model, newdata = test_matrix)
conf_matrix_rf <- table(Predicted = rf_preds, Actual = test_data$Winner)
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)

rf_probs <- predict(rf_model, newdata = test_matrix, type = "prob")[,2]
roc_rf <- roc(test_data$Winner, rf_probs)
auc_rf <- auc(roc_rf)

# Comparación gráfica inicial
plot(roc_log, col = "blue", main = "Curvas ROC")
lines(roc_rf, col = "red")
legend("bottomright", legend = c("Logística", "Random Forest"), col = c("blue", "red"), lwd = 2)

print(paste("Precisión Regresión Logística:", round(accuracy_log, 4)))
print(paste("AUC Regresión Logística:", round(auc_log, 4)))
print(paste("Precisión Random Forest:", round(accuracy_rf, 4)))
print(paste("AUC Random Forest:", round(auc_rf, 4)))

# 7. OPTIMIZACIÓN RANDOM FOREST (caret) ----
train_data$Winner <- factor(train_data$Winner, levels = c(0, 1), labels = c("No", "Yes"))
test_data$Winner <- factor(test_data$Winner, levels = c(0, 1), labels = c("No", "Yes"))

tune_grid <- expand.grid(mtry = c(2, 4, 6, 8))
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE)

set.seed(123)
rf_tuned <- train(
  Winner ~ ., data = train_data,
  method = "rf", metric = "ROC",
  trControl = control, tuneGrid = tune_grid,
  ntree = 100
)

print(rf_tuned)
plot(rf_tuned)

# 8. OPTIMIZACIÓN REGRESIÓN LOGÍSTICA REGULARIZADA ----
tune_grid_log <- expand.grid(
  alpha = c(0, 0.5, 1),
  lambda = 10^seq(-4, 1, length = 10)
)

control_log <- trainControl(
  method = "cv", number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)

set.seed(123)
logit_tuned <- train(
  Winner ~ ., data = train_data,
  method = "glmnet",
  trControl = control_log,
  tuneGrid = tune_grid_log,
  metric = "ROC"
)

print(logit_tuned)
plot(logit_tuned)

# ------------------ GLM clásico ------------------
conf_matrix_log <- table(Predicted = pred_classes, Actual = test_data$Winner)
accuracy_log <- sum(diag(conf_matrix_log)) / sum(conf_matrix_log)
roc_log <- roc(test_data$Winner, pred_probs)
auc_log <- auc(roc_log)

# ------------------ Random Forest (ajustado) ------------------
rf_preds <- predict(rf_tuned, newdata = test_data)
rf_probs <- predict(rf_tuned, newdata = test_data, type = "prob")[, "Yes"]
conf_matrix_rf <- table(Predicted = rf_preds, Actual = test_data$Winner)
accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
roc_rf <- roc(test_data$Winner, rf_probs)
auc_rf <- auc(roc_rf)

# ------------------ Regresión Logística Regularizada (GLMNET) ------------------
logit_probs <- predict(logit_tuned, newdata = test_data, type = "prob")[, "Yes"]
logit_preds <- ifelse(logit_probs > 0.5, "Yes", "No")
conf_matrix_logit <- table(Predicted = logit_preds, Actual = test_data$Winner)
accuracy_logit <- sum(diag(conf_matrix_logit)) / sum(conf_matrix_logit)
roc_logit <- roc(test_data$Winner, logit_probs)
auc_logit <- auc(roc_logit)

# ------------------ Comparación de Resultados ------------------
cat("\n--- Evaluación de Modelos ---\n")
cat("Regresión Logística (clásica):\n")
cat(paste("  - Precisión:", round(accuracy_log, 4), "\n"))
cat(paste("  - AUC:", round(auc_log, 4), "\n\n"))

cat("Random Forest (tuned):\n")
cat(paste("  - Precisión:", round(accuracy_rf, 4), "\n"))
cat(paste("  - AUC:", round(auc_rf, 4), "\n\n"))

cat("Regresión Logística Regularizada (GLMNET):\n")
cat(paste("  - Precisión:", round(accuracy_logit, 4), "\n"))
cat(paste("  - AUC:", round(auc_logit, 4), "\n"))

# 9. EVALUACIÓN FINAL Y COMPARACIÓN ROC ----
plot(roc_logit, col = "blue", main = "Curvas ROC comparativas")
lines(roc_rf, col = "red")
lines(roc_log, col = "green")
legend("bottomright",
       legend = c(
         paste("Logística (clásica) - AUC =", round(auc_log, 3)),
         paste("Random Forest - AUC =", round(auc_rf, 3)),
         paste("GLMNET - AUC =", round(auc_logit, 3))
       ),
       col = c("green", "red", "blue"),
       lwd = 2)


# 10. GUARDAR MODELO FINAL ----
saveRDS(rf_tuned, "modelo_rf_final.rds")

# Cargar modelo entrenado
rf_model_loaded <- readRDS("modelo_rf_final.rds")
