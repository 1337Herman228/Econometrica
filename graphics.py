import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import np
import scipy.stats as stats
from scipy.stats import skew, kurtosis
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.stats.diagnostic import het_white

df = pd.read_csv("E:/VsCode/Python/Econometrica_kursach/EXTENDED_PC.csv") #читаем файл
df.head()
print(df)

# Вычисляем среднее значение
mean_price = df['Price'].mean()

# Вычисляем медиану
median_price = df['Price'].median()

# Вычисляем нижний квартиль (25-й процентиль)
lower_quartile = df['Price'].quantile(0.25)

# Вычисляем верхний квартиль (75-й процентиль)
upper_quartile = df['Price'].quantile(0.75)

print("Среднее значение: ", mean_price)
print("Медиана: ", median_price)
print("Нижний квартиль: ", lower_quartile)
print("Верхний квартиль: ", upper_quartile)

####################### Гистограмма аномальных значений #########################

Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_df = df[(df['Price'] >= lower_bound) & (df['Price'] <= upper_bound)]

def find_and_plot_anomalies(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    
    # Построение гистограммы с учетом аномальных значений
    plt.figure(figsize=(12, 6))
    plt.hist(data[feature], bins=20, color='skyblue', alpha=0.7, label='Все данные')
    plt.hist(anomalies[feature], bins=20, color='red', alpha=0.7, label='Аномальные значения')
    plt.xlabel(feature)
    plt.ylabel('Количество значений')
    plt.title('Гистограмма значений и аномальных значений признака ' + feature)
    # Добавление медианы и среднего значения на график
    median_price = filtered_df[feature].median()
    mean_price = filtered_df[feature].mean()
    plt.axvline(median_price, color='blue', linestyle='-', label='Медиана')
    plt.axvline(mean_price, color='red', linestyle='--', label='Среднее значение')
    plt.legend()
    plt.show()

# find_and_plot_anomalies(filtered_df, 'Price')
# find_and_plot_anomalies(filtered_df, 'RAM_Frequency')
# find_and_plot_anomalies(filtered_df, 'RAM_Volume')
# find_and_plot_anomalies(filtered_df, 'CPU_Performance_coef')
find_and_plot_anomalies(filtered_df, 'GPU_VRAM')


#######################################################################


############################# Ящики с усами ##############################

def makeBoxplot(param, color = '#536ef5'):
    ax = sns.boxplot(y = param, data = filtered_df, color=color, width = .1) # ящик с усами (распределение и выбросы)
    ax.set_ylabel(param)
    plt.show()

# makeBoxplot("Price", "red")
# makeBoxplot("RAM_Frequency", "blue")
# makeBoxplot("RAM_Volume", "green")
# makeBoxplot("CPU_Performance_coef", "orange")
makeBoxplot("GPU_VRAM", "purple")

#######################################################################

data_skewness = skew(filtered_df['Price'])

# Вычисляем коэффициент эксцесса
data_kurtosis = kurtosis(filtered_df['Price'], bias= False)

print('Показатель асимметрии:', data_skewness)
print('Коэффициент эксцесса:', data_kurtosis)

selected_columns = [
                    'Price',
                    'RAM_DDR_Type',
                    'RAM_Price',
                    'RAM_Frequency', 
                    'RAM_Volume', 
                    'CPU_Price',
                    'CPU_Performance_coef',
                    'CPU_Core_count', 
                    'CPU_SMT', 
                    'CPU_Boost_clock', 
                    'CPU_TDP', 
                    'GPU_Price',
                    'GPU_Performance_coef',
                    'GPU_VRAM', 
                    'GPU_Boost_clock', 
                    # 'Storage_Capacity',
                    ]
selected_df = filtered_df[selected_columns]
correlation_matrix = selected_df.corr()


####################### Матрица корреляции #########################

plt.matshow(correlation_matrix, cmap='coolwarm')
plt.colorbar()
for (i, j), val in np.ndenumerate(correlation_matrix):
    plt.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.show()

#######################################################################

####################### Диаграммы рассеивания #########################
def find_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def make_scatter_plot(x, y, xlabel, ylabel):
    x_df = filtered_df[x]
    y_df = filtered_df[y]
    plt.scatter(x_df, y_df)

    #  Помечаем выбросы
    X_lower_bound, X_upper_bound = find_outliers(x_df)
    Y_lower_bound, Y_upper_bound = find_outliers(y_df)
    outliers = df[(df[x] < X_lower_bound) | (df[x] > X_upper_bound) | (df[y] < Y_lower_bound) | (df[y] > Y_upper_bound)]
    plt.scatter(outliers[x], outliers[y], color='red', label='Выбросы')
    plt.legend()
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Диаграмма рассеяния:' + xlabel + ' и ' + ylabel)
    plt.show()

# make_scatter_plot('Price', 'RAM_Frequency', 'Цена ПК', 'Частота оперативной памяти (Мгц)')
# make_scatter_plot('Price', 'RAM_Volume', 'Цена ПК', 'Объем оперативной памяти (Гб)')
# make_scatter_plot('Price', 'CPU_Performance_coef', 'Цена ПК', 'Коэффициент производительности процессора')
make_scatter_plot('Price', 'GPU_VRAM', 'Цена ПК', 'Объем видеопамяти (Гб)')

#######################################################################

##################### Построение и обучение модели ###########################

X = filtered_df[[
                # 'RAM_DDR_Type', 
                # 'RAM_Price',
                'RAM_Frequency', 
                'RAM_Volume', 
                # 'CPU_Price',
                'CPU_Performance_coef',
                # 'CPU_Core_count', 
                # 'CPU_SMT',
                # 'CPU_Boost_clock', 
                # 'CPU_TDP', 
                # 'GPU_Price',
                # 'GPU_Performance_coef',
                'GPU_VRAM', 
                # 'GPU_Boost_clock', 
                # 'Storage_Capacity',
                ]]
X = sm.add_constant(X)
Y = filtered_df['Price']
# Создание модели линейной регрессии
model = sm.OLS(Y, X)
# Обучение модели
result = model.fit()
# Вывод результатов
print(result.summary())

# Получение остатков  ///////////////////////////////////////////////////////////////////////////////////
residuals = result.resid
for i in residuals.index:
    if residuals[i] > 100:
        print(i)
        print(residuals[i])
# print(residuals)

##################### График остатков ###########################

plt.figure(figsize=(10, 6))
plt.scatter(result.fittedvalues, residuals, color='skyblue')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('График остатков')
plt.xlabel('Price')
plt.ylabel('Значения остатков')
plt.legend()
plt.show()

##################################################################################

##################### Проверка остатков на нормальное распределение ###########################

# Выполнение теста Жака-Бера на нормальность остатков
jb_test = sm.stats.jarque_bera(residuals)
jb_prob = jb_test[1]

print("p-value:", jb_test[1])
if jb_prob > 0.05:
    print("Распределение остатков НОРМАЛЬНОЕ")
else:
    print("Распределение остатков не нормальное")

# Построение гистограммы распределения остатков
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=100, color='skyblue', alpha=0.7)
plt.title('Гистограмма распределения остатков')
plt.xlabel('Значение остатков')
plt.ylabel('Количество')
plt.legend(['Jarque-Bera : ' + str(jb_test[0].round(3)) + '\n' + 'Probability : ' + str(jb_prob)], loc='upper right')

plt.show()

########################################################################################################

##################### Проверка остатков автокорреляцию тестом Дарбина-Уотсона ###########################

# Расчет статистики Дарбина-Уотсона
dw_test = sm.stats.stattools.durbin_watson(residuals)

dl = 1.57
du = 1.78

# Задание координат зон
zone_colors = ['#F08080', '#808080', '#98FB98', '#808080', '#F08080']
zone_values = [0, dl, du, 4-du, 4-dl, 4]

# Построение графика с зонами
plt.figure(figsize=(8, 12))

for i in range(len(zone_values)-1):
    label = ''
    if i==0 : label = 'Положительная автокорреляция (Отклонение H0)'
    if i==1 or i==3 : label = 'Неопределенность'
    if i==2 : label = 'Автокорреляции нет (H0)'
    if i==4 : label = 'Отрицательная автокорреляция (Отклонение H0)'
    plt.axvspan(zone_values[i], zone_values[i+1], color=zone_colors[i], alpha=0.5, label=label)

plt.title('Тест Дарбина-Уотсона')
plt.xlim(0, 4)  # Установка пределов оси X
plt.axvline(x=dw_test, color='r', linestyle='--', label='DW: ' + str(dw_test.round(3)))
plt.legend()

plt.show()

########################################################################################################


##################### Тест Уайта на гетероскедастичность ################

# Выполняем тест Уайта на гетероскедастичность
white_test = het_white(residuals, X)

# Выводим результаты теста
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
results = pd.DataFrame(list(white_test), index=labels, columns=['Test Results'])

print(results)
if white_test[3] < 0.05:
    print("Остатков гетероскедастичны")
else:
    print("Остатки ГОМОСКЕДАСТИЧНЫ")

########################################################################

##################### Точечный прогноз индивидуального значения показателя ################

# Создаем график с прогнозами и фактическими значениями
plt.figure(figsize=(10, 6))

plt.plot(Y, label='Фактическое Y', marker='x')
plt.plot(result.fittedvalues, label='Предсказанное Y', marker='o')
plt.xlabel('Наблюдение')
plt.ylabel('Значение')
plt.title('Точечный прогноз индивидуального значения показателя')
plt.legend()
plt.grid(True)
plt.show()

#######################################################################################


################ Доверительный интервал для прогноза математического ожидания результирующего показателя ##############################

# # Предположим, что model - ваша модель линейной регрессии, и X - независимая переменная для прогноза
# # Получаем предсказания и доверительные интервалы
# predictions = result.get_prediction()
# predicted_mean = predictions.predicted_mean
# confidence_interval = result.conf_int(0.05, None)

# print(Y)
# # Преобразовать Series Y в DataFrame
# df = pd.DataFrame(Y)

# # Выбрать первый столбец
# index_column = df.reset_index().iloc[:, 0]
# # print(index_column)
# # print(confidence_interval.iloc[:, 0][5])
# # print(confidence_interval.iloc[:, 1][5])

# plt.fill_between(index_column, Y + confidence_interval.iloc[:, 0][5], Y+ confidence_interval.iloc[:, 1][5], color='r', label='Доверительный интервал')
# plt.scatter(index_column, Y, label='Фактическое Y', marker='x')
# plt.plot(result.fittedvalues, label='Предсказанное Y', color='black')
# plt.xlabel('Наблюдение')
# plt.ylabel('Значение')
# plt.title('Точечный прогноз индивидуального значения показателя')
# plt.legend()
# plt.grid(True)
# plt.show()

###########################################################################################################

# ----------------------------------

# # Создаем данные для графика
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)  # первая кривая
# y2 = np.cos(x)  # вторая кривая

# # Заполняем область между кривыми
# plt.fill_between(x, y1, y2, color='lightblue', alpha=0.5, label='Filled Area')

# # Отображаем кривые на графике
# plt.plot(x, y1, label='Sin(x)')
# plt.plot(x, y2, label='Cos(x)')

# # Добавляем легенду, оси и заголовок
# plt.legend()
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Fill Between Example')

# # Показываем график
# plt.show()