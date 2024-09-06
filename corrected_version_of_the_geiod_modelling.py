import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# Load CSV data
def load_data(file_path):
    data = pd.read_csv(file_path)
    x = data[['LAT', 'LON', 'h']]
    y = data['Difference in N']
    return x, y, data


# Split data into training and testing sets
def split_data(x, y, test_size=0.2, random_state=42):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


# Hyperparameter tuning for Random Forest Regressor
def hyperparameter_tuning(x_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 5, 10],
    }
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# Evaluate model on test data
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_test, y_pred, rmse, mae, r2


# Polynomial Regression
def polynomial_regression(x_train, y_train, x_test, degree=5):
    poly_feature = PolynomialFeatures(degree=degree)
    x_train_poly = poly_feature.fit_transform(x_train)
    model = RandomForestRegressor()
    model.fit(x_train_poly, y_train)
    x_test_poly = poly_feature.transform(x_test)
    return model.predict(x_test_poly)


# Group Method for Data Handling
def group_method_data_handling(data, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    data['group'] = kmeans.fit_predict(data[['LAT', 'LON', 'h']])
    y_test = np.array([])
    y_pred = np.array([])

    for group in data['group'].unique():
        group_data = data[data['group'] == group]
        x_train, x_test, y_train, y_test_group = train_test_split(group_data[['LAT', 'LON', 'h']],
                                                                  group_data['Difference in N'], test_size=0.2,
                                                                  random_state=42)
        model = RandomForestRegressor()
        model.fit(x_train, y_train)
        y_test = np.concatenate((y_test, y_test_group))
        y_pred = np.concatenate((y_pred, model.predict(x_test)))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_test, y_pred, rmse, mae, r2


# M5 Model Tree
def m5_modelling_tree(data, num_clusters=5):
    return group_method_data_handling(data, num_clusters)


# Gaussian Process Regression
def gaussian_process_regression(data, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    data['group'] = kmeans.fit_predict(data[['LAT', 'LON', 'h']])
    y_test = np.array([])
    y_pred = np.array([])

    for group in data['group'].unique():
        group_data = data[data['group'] == group]
        x_train, x_test, y_train, y_test_group = train_test_split(group_data[['LAT', 'LON', 'h']],
                                                                  group_data['Difference in N'], test_size=0.2,
                                                                  random_state=42)

        kernel = Matern(nu=1.5)
        model = GaussianProcessRegressor(kernel=kernel)
        model.fit(x_train, y_train)
        y_test = np.concatenate((y_test, y_test_group))
        y_pred = np.concatenate((y_pred, model.predict(x_test)))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_test, y_pred, rmse, mae, r2


# Gradient Boosting Regression
def gradient_boosting_regression(data, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    data['group'] = kmeans.fit_predict(data[['LAT', 'LON', 'h']])
    y_test = np.array([])
    y_pred = np.array([])

    for group in data['group'].unique():
        group_data = data[data['group'] == group]
        x_train, x_test, y_train, y_test_group = train_test_split(group_data[['LAT', 'LON', 'h']],
                                                                  group_data['Difference in N'], test_size=0.2,
                                                                  random_state=42)

        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        model.fit(x_train, y_train)
        y_test = np.concatenate((y_test, y_test_group))
        y_pred = np.concatenate((y_pred, model.predict(x_test)))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_test, y_pred, rmse, mae, r2


# Resemblance Learning
def resemblance_learning(data, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    data['group'] = kmeans.fit_predict(data[['LAT', 'LON', 'h']])
    y_test = np.array([])
    y_pred = np.array([])

    for group in data['group'].unique():
        group_data = data[data['group'] == group]
        x_train, x_test, y_train, y_test_group = train_test_split(group_data[['LAT', 'LON', 'h']],
                                                                  group_data['Difference in N'], test_size=0.2,
                                                                  random_state=42)

        model = KNeighborsRegressor(n_neighbors=1)
        model.fit(x_train, y_train)
        y_test = np.concatenate((y_test, y_test_group))
        y_pred = np.concatenate((y_pred, model.predict(x_test)))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_test, y_pred, rmse, mae, r2


# Support Vector Regression
def support_vector_regression(data, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters)
    data['group'] = kmeans.fit_predict(data[['LAT', 'LON', 'h']])
    y_test = np.array([])
    y_pred = np.array([])

    for group in data['group'].unique():
        group_data = data[data['group'] == group]
        x_train, x_test, y_train, y_test_group = train_test_split(group_data[['LAT', 'LON', 'h']],
                                                                  group_data['Difference in N'], test_size=0.2,
                                                                  random_state=42)

        model = SVR(kernel='rbf', C=5, gamma=0.1)
        model.fit(x_train, y_train)
        y_test = np.concatenate((y_test, y_test_group))
        y_pred = np.concatenate((y_pred, model.predict(x_test)))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return y_test, y_pred, rmse, mae, r2


def calculate_correlation(y_test, y_pred):
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    return corr


def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.figure(figsize=(15, 10),fontsize=20)#10,6
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted for {model_name}')
    plt.show()


def save_actual_predicted(y_test, y_pred, model_name):
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    results.to_csv(f'{model_name}_results.csv', index=False)


def main():
    file_path = r"C:\Users\USER\Desktop\VICENTIA\hybrid_geoid_sample_data.csv"
    x, y, data = load_data(file_path)
    x_train, x_test, y_train, y_test = split_data(x, y)

    # Random Forest
    rf_model, rf_best_params = hyperparameter_tuning(x_train, y_train)
    y_test_rf, y_pred_rf, rmse_rf, mae_rf, r2_rf = evaluate_model(rf_model, x_test, y_test)
    corr_rf = calculate_correlation(y_test_rf, y_pred_rf)

    # Polynomial Regression
    y_pred_poly = polynomial_regression(x_train, y_train, x_test)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    mae_poly = mean_absolute_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    corr_poly = calculate_correlation(y_test, y_pred_poly)

    # M5 Modelling Tree
    y_test_m5, y_pred_m5, rmse_m5, mae_m5, r2_m5 = m5_modelling_tree(data)
    corr_m5 = calculate_correlation(y_test_m5, y_pred_m5)

    # Gaussian Process Regression
    y_test_gp, y_pred_gp, rmse_gp, mae_gp, r2_gp = gaussian_process_regression(data)
    corr_gp = calculate_correlation(y_test_gp, y_pred_gp)

    # Gradient Boosting Regression
    y_test_gb, y_pred_gb, rmse_gb, mae_gb, r2_gb = gradient_boosting_regression(data)
    corr_gb = calculate_correlation(y_test_gb, y_pred_gb)

    # Resemblance Learning
    y_test_rl, y_pred_rl, rmse_rl, mae_rl, r2_rl = resemblance_learning(data)
    corr_rl = calculate_correlation(y_test_rl, y_pred_rl)

    # Support Vector Regression
    y_test_svr, y_pred_svr, rmse_svr, mae_svr, r2_svr = support_vector_regression(data)
    corr_svr = calculate_correlation(y_test_svr, y_pred_svr)

    # Plotting
    models = ['RANDOM_FOREST', 'POLYNOMIAL', 'M5_MODELLING', 'GAUSSIAN_P ',  'GRADIENT_B',
              'RESEMBLANCE', 'SUPPORT_VECTOR ']
    predictions = [y_pred_rf, y_pred_poly, y_pred_m5, y_pred_gp, y_pred_gb, y_pred_rl, y_pred_svr]
    actuals = [y_test_rf, y_test, y_test_m5, y_test_gp, y_test_gb, y_test_rl, y_test_svr]
    rmses = [rmse_rf, rmse_poly, rmse_m5, rmse_gp, rmse_gb, rmse_rl, rmse_svr]
    maes = [mae_rf, mae_poly, mae_m5, mae_gp, mae_gb, mae_rl, mae_svr]
    r2s = [r2_rf, r2_poly, r2_m5, r2_gp, r2_gb, r2_rl, r2_svr]
    corrs = [corr_rf, corr_poly, corr_m5, corr_gp, corr_gb, corr_rl, corr_svr]

    fig, axes = plt.subplots(4, 2, figsize=(70,60)) #2024
    fig.suptitle('ACTUAL VS PREDICTED PLOTS FOR THE VARIOUS MODELS', fontsize=110)
    axes = axes.flatten()

    for i, model in enumerate(models):
        # axes[i].scatter(actuals[i], predictions[i], alpha=1)
        axes[i].scatter(actuals[i], predictions[i],s=1000)
        axes[i].plot([actuals[i].min(), actuals[i].max()], [actuals[i].min(), actuals[i].max()], 'r--', lw=2)
        axes[i].set_xlabel('ACTUAL',fontsize=70)
        axes[i].set_ylabel('PREDICTED',fontsize=70)
        axes[i].set_title(f'ACTUAL VS PREDICTED FOR  {model}',fontsize=100)
        axes[i].tick_params(axis='x', labelsize=60)  # increase x-axis value size
        axes[i].tick_params(axis='y', labelsize=60)




    if len(models) % 2 != 0:
        fig.delaxes(axes[-1])
    plt.subplots_adjust(wspace=0.5,hspace=0.7)
    plt.tick_params(axis='x')
    # plt.tick_params(axis='y', labelsize=30)

    plt.savefig(r"C:\Users\USER\Desktop\VICENTIA\COODING_FILE\DAKR_pwv_daily.png")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(60, 40))   #15,17
    fig.suptitle('ACCURACIES PLOTS FOR THE VARIOUS MODELS', fontsize=100)

    axes[0, 0].bar(models, rmses, color='skyblue')
    axes[0, 0].set_title('RMSE COMPARISON',fontsize=70)
    axes[0, 0].set_ylabel('RMSE',fontsize=70)
    axes[0, 0].tick_params(axis='x', rotation=90,labelsize=40)
    axes[0, 0].tick_params(axis='y', labelsize=40)
    for i, value in enumerate(rmses):
        axes[0, 0].text(i, value + 0.01, str(round(value, 4)), ha='center',fontsize=40)

    axes[0, 1].bar(models, maes, color='salmon')
    axes[0, 1].set_title('MAE COMPARISON',fontsize=70)
    axes[0, 1].set_ylabel('MAE',fontsize=70)
    axes[0, 1].tick_params(axis='x', rotation=90,labelsize=40)
    axes[0, 1].tick_params(axis='y',  labelsize=40)
    for i, value in enumerate(maes):
        axes[0, 1].text(i, value + 0.01, str(round(value, 4)), ha='center',fontsize=40)

    axes[1, 0].bar(models, r2s, color='lightgreen')
    axes[1, 0].set_title('R² COMPARISON',fontsize=70)
    axes[1, 0].set_ylabel('R² SCORE',fontsize=70)
    axes[1, 0].tick_params(axis='x', rotation=90,labelsize=40)
    axes[1, 0].tick_params(axis='y', labelsize=40)
    for i, value in enumerate(r2s):
        axes[1, 0].text(i, value + 0.01, str(round(value, 4)), ha='center',fontsize=40)

    axes[1, 1].bar(models, corrs, color='lightcoral')
    axes[1, 1].set_title('CORRELATION COMPARISON',fontsize=70)
    axes[1, 1].set_ylabel('CORRELATION',fontsize=70)
    axes[1, 1].tick_params(axis='x', rotation=90,labelsize=40)
    axes[1, 1].tick_params(axis='y',  labelsize=40)
    for i, value in enumerate(corrs):
        axes[1, 1].text(i, value + 0.01, str(round(value, 4)), ha='center',fontsize=40)

    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    # plt.tick_params(axis='x', labelsize=25)

    plt.savefig(r"C:\Users\USER\Desktop\VICENTIA\hybrid_geoid_sample_data.png")
    plt.close()

    # # plt.tight_layout()
    # plt.show()

    # Save results
    save_actual_predicted(y_test_rf, y_pred_rf, 'Random_Forest')
    save_actual_predicted(y_test, y_pred_poly, 'Polynomial_Regression')
    save_actual_predicted(y_test_m5, y_pred_m5, 'M5_Modelling_Tree')
    save_actual_predicted(y_test_gp, y_pred_gp, 'Gaussian_Process_Regression')
    save_actual_predicted(y_test_gb, y_pred_gb, 'Gradient_Boosting_Regression')
    save_actual_predicted(y_test_rl, y_pred_rl, 'Resemblance_Learning')
    save_actual_predicted(y_test_svr, y_pred_svr, 'Support_Vector_Regression')


if __name__ == '__main__':
    main()
