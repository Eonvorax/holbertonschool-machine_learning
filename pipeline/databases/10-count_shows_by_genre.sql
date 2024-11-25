-- lists all genres and the number of shows linked to each
SELECT tg.name AS genre, COUNT(tsg.genre_id) AS number_of_shows
FROM tv_genres tg, tv_show_genres tsg
WHERE tg.id = tsg.genre_id
GROUP BY genre
ORDER BY number_of_shows DESC;
