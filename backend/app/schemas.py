from pydantic import BaseModel
from datetime import date

class RealEstateFeatures(BaseModel):
    # Физические характеристики
    region: str                       # регион (категория)
    area_sqm: float                   # общая площадь, м²
    floor: int                        # этаж
    total_floors: int                 # общее число этажей
    year_built: int                   # год постройки
    house_type: str                   # новостройка / вторичка
    material: str                     # материал стен
    renovation: str                   # тип ремонта
    ceiling_height_m: float           # высота потолков, м
    layout: str                       # планировка
    has_balcony: bool                 # наличие балкона/лоджии
    view: str                         # вид из окон
    has_parking: bool                 # наличие парковки

    # Локальные соц.-экономические параметры
    distance_to_center_km: float      # до центра города, км
    distance_to_metro_km: float       # до метро/остановки, км
    crime_rate: float                 # уровень преступности
    infra_objects: int                # кол-во соц. объектов (школы, поликлиники и т.п.)
    building_density: float           # плотность застройки
    income_level: float               # средний доход в районе
    rental_share: float               # доля арендного жилья (0–1)
    new_housing_delivery: int         # введено домов, шт

    # Макроэкономика
    dollar_rate: float                # курс USD/RUB
    euro_rate: float                  # курс EUR/RUB
    key_rate: float                   # ключевая ставка ЦБ, %
    cpi: float                        # индекс потребительских цен, %
    pmi: float                        # индекс деловой активности (Manufacturing PMI)
    moex_index: float                 # индекс МосБиржи
    oil_price: float                  # Brent, USD/баррель
    unemployment: float               # уровень безработицы, %
    gdp_growth: float                 # темп прироста ВВП, %
    mortgage_rate: float              # средняя ипотечная ставка, %
    consumer_confidence: float        # индекс потребительского доверия
    households_debt_to_gdp: float     # закредитованность населения, % от ВВП
