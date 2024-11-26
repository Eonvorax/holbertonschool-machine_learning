-- lists all bands with 'Glam rock' as their main style, ranked by their longevity (using attributes 'formed' and 'split')
SELECT
    band_name,
    CASE
        WHEN split IS NULL THEN 2020 - formed
        ELSE split - formed
    END AS lifespan
FROM
    metal_bands
WHERE
    style LIKE '%Glam rock%'
ORDER BY
    lifespan DESC;
