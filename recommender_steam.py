import pandas as pd
import numpy as np
from sklearn import preprocessing
import difflib
import os

user_id = input("Kullanıcı Id'nizi Giriniz:\n")
print(user_id)
# Veriyi pandas ile okuyoruz. Verinin başlık satırı yok ise names parametresinde belirterek
# başlıkları satır olarak verisetinin başına ekliyoruz.
rating = pd.read_csv('Veriseti/steam-200k.csv', names=["userId", "gameName", 
                                                     "purchase", "hoursPlayed", "null"])
# şimdilik purchase keywordünü içeren satırları kullanmayıp sadece play keywordünü içeren
# satırları filtreden geçirerek verisetimizi güncelliyoruz.
rating = rating[rating.purchase == 'play']
# indexleri resetliyoruz.
rating = rating.reset_index(drop=True)
# normalizasyon işlemi için önce hoursPlayed sütununu çekiyoruz.
x = [rating['hoursPlayed']]
x = np.reshape(x, (len(x[0]),1))
# normalizasyon işlemini gerçekleştirecek olan preprocessing kütüphanesini çağırıyoruz.
min_max_scaler = preprocessing.MinMaxScaler()
# normalizasyon işlemini gerçekleştiriyoruz.
x_scaled = min_max_scaler.fit_transform(x)
x_scaled = x_scaled.flatten()
# normalize edilmiş veriyi yeniden DataFrame'deki hoursPlayed sütununa eşitleyerek
# bu sütunu normalize edilmiş olarak güncelliyoruz.
rating['hoursPlayed'] = x_scaled

# Ham verisetini okutturuyoruz.
'''print(rating)

# =============================================================================
## Oyunların adlarını çektik.
# game = rating.gameName
## Duplicateleri yok ettik.
# game = game.drop_duplicates()
# Incremental olacak şekilde index sütununu gameId sütununa eşitledik ve gameId sütunumuz
# index sütununa eş değer oldu.
# game['gameId'] = game.index

# Elde ettiğimiz oyun verisetini okutturuyoruz
# print(game)

# Kullanıcıların id'lerini çektik.
# user = rating.userId
# Duplicateleri yok ettik.
# user = user.drop_duplicates()
# Id sütunu oluşturabilmek için indexleri resetledik.
user = user.reset_index()
# İlk sütunumuzu userId sütunu olarak isimlendirdik.
user.rename(columns={('index'): ('userId')}, inplace=True)
# Duplicate olan userId sütunlarını yok ettik.
user = user.loc[:,~user.columns.duplicated()]
# Incremental olacak şekilde index sütununu userId sütununa eşitledik ve userId sütunumuz
# index sütununa eş değer oldu. 
user['userId'] = user.index

# Elde ettiğimiz user verisetini okutturuyoruz.
print(user)

# rating ve game verisetini merge ettik böylece bir verisetinde gameId, gameName, userId,
# purchase, hoursPlayed gibi değişkenleri tek bir verisetinde topladık.
data = pd.merge(user, rating)
'''
# userId - gameName - hoursPlayed kolonları içerecek şekilde tablosunu oluşturduk.
data = rating[['userId', 'gameName', 'hoursPlayed']]
# ilgili tabloyu okuttuk.
print(data)


# vg_sales verisetinden elde ettiğimiz benzer isimli oyunları çekiyoruz.
similar_named_games = pd.read_csv('Veriseti/close_matched_games.csv', names=["gameNameAfter", "gameName"], delimiter=';')
# oyunları yeni verisetindeki oyunlar ile güncelleyebilmek için merge ediyoruz.
data = pd.merge(data, similar_named_games)
# oyun adları sütunumuzu güncelliyoruz.
data['gameName'] = data['gameNameAfter']
# artık gerek olmayan gameNameAfter sütununu siliyoruz.
del data['gameNameAfter']
# son halini okutuyoruz.
print(data)

# vg_sales verisetini content based filtreleme için dosyadan okuyoruz.
vg_sales = pd.read_csv('Veriseti/vgsales.csv')
# Name sütununun adını gameName olarak değiştiriyoruz.
vg_sales = vg_sales.rename(columns={'Name': 'gameName'})
# Daha sonra oyunların adlarının yer aldığı sütunu ve türlerini çekiyoruz.
data2 = vg_sales[['gameName', 'Genre']]

# elde ettiğimiz dataFrame verisini okutuyoruz.
print(data2)


# son olarak verisetimize oyun türlerini dahil edebilmek için data2 veriseti ile
# bir merge işlemi daha gerçekleştiriyoruz.
data = pd.merge(data, data2, on='gameName', how='left')
# duplicate'lerden arındırıp güncel halini elde ediyoruz.
data.drop_duplicates(subset=['userId', 'gameName'],
                     keep = 'last', inplace = True)
# son halini okutuyoruz.
print(data)


# =============================================================================
# # Öncelikle iki verisetinde benzer isme sahip olan oyunlar çekebilmek için close_matched_games
# # adlı bir liste değişkeni oluşturduk.
# close_matched_games = np.asarray([])
# # Steam verisetindeki her oyunun vg_sales verisetindeki oyunlardan benzer isme sahip olanlarını
# # difflib kütüphanesi ile hesaplatarak listemize ekledik.
# i = 0
# for gameName in game['gameName']:
#     close_matched_games = np.append(close_matched_games, values = ([gameName], difflib.get_close_matches(gameName, data2['gameName'])[:1] or None))
#     i = i + 1
#     if i > 10:
#         break
#     
# print(close_matched_games)
# 
# # Son olarak elde ettiğimiz sonuçları csv olarak kaydettik.
# # NOT: Bu işlemi yapmamız ilk verisetindeki oyunlar ile ikinci verisetindeki oyunları
# # eşleştirip türlerini yeni bir DataFrame içinde ortaya koyabilmek için çok önemliydi.
# 
# np.savetxt("deneme.csv", close_matched_games, delimiter=",", fmt='%s')
# =============================================================================
# İlgili kullanıcı için oynanılan oyunları, kategorileri ile birlikte getiriyoruz.
category = data[data['userId'] == int(user_id)]
# çektiğimiz oyunları okutuyoruz.
print(category)
# Kullanıcının oynadığı oyunların kategorilere göre saatleri hesaplanılarak en çok hangi kategoride oynadığını bulmak için
# kategoriye göre toplayıp sıralamasını sağlıyoruz.
hours = category.groupby('Genre').sum('hoursPlayed').sort_values('hoursPlayed', ascending=False)
# Kategorilere göre oyun sürelerini sıralı bir şekilde yazdırıyoruz.
print(hours)
# En çok oynanılan kategoriyi buluyoruz.
temp = category['Genre']
temp = temp.drop_duplicates()
# En çok oynadığımız kategori ismini yazdırıyoruz.
print(temp.values[0])
# Oyun listesini en çok oynadığımız oyun kategorisine göre filtreliyoruz.
category = category[category['Genre'] == temp.values[0]]
# Filtrelenen listeyi yazdırıyoruz.
print(category)
# En çok oynadığımız kategorideki en çok süre harcadığımız oyunu seçiyoruz.
most_played = category.head(1)
# Seçtiğimiz oyunu yazdırıyoruz.
print(most_played)
# Seçilen oyunun ismini alıyoruz.
game_name = most_played['gameName'].values[0]
# Seçilen oyunun ismini yazdırıyoruz.
print(game_name)
# kullanıcılar satır ve oyunlar sütun olacak şekilde kullanıcıların oyunları kaç saat
# oynadığını içeren, öneri sistemleri algoritmalarının kolay bir şekilde işletilebileceği
# bir pivot tablosu oluşturduk.
# =============================================================================
pivot_table = data[data['Genre'] == temp.values[0]].pivot_table(index = ["userId"], columns = ["gameName"], values = "hoursPlayed")
# Oluşan pivot table'ı yazdırıyoruz.
print(pivot_table)
# 
# # Herhangi bir oyunun diğer oyunlar ile olan benzerlik durumunu ölçebilmek için
# # pivot tablosundan bir oyun sütununu çekiyoruz.
game_played = pivot_table[game_name]
print(game_played)
# # corrwith fonksiyonu ile bu oyunun diğer oyunlar ile olan benzerliğini hesaplatıyoruz.
similarity_with_other_games = pivot_table.corrwith(game_played)
# # benzerlik oranı en yüksek olan oyunlardan en düşük olan oyunlara olacak şekilde benzerlik
# # oranı listesini yeniden sıralıyoruz.
similarity_with_other_games = similarity_with_other_games.sort_values(ascending=False)
# # Listeden en çok benzeyen ilk 5 oyunu çekiyoruz ve konsol penceresine yazdırtıyoruz.
similarity_with_other_games = similarity_with_other_games.drop(category['gameName'])
print(similarity_with_other_games.head(50))
# =============================================================================








    

