import numpy as np
velocity = [8.783993497553004, -876939.4829423582, -839886.0676025964, 8.917130009974578, 8.815464729068541, 8.693624072824228, 8.915620588620511, -821114.7473695726, 8.942623322204966, 8.969921090792692, 8.97612292545698, 9.04485981256023, 8.85779666633905, 8.869987616601268, 8.889177162163058, -891905.0726821837, 8.870169067362738, -855158.0730186608, -820320.9779584906, 8.83107809370345, 8.756772168357886, 8.946230812581446, 8.94928382849937, 8.952814903367885, 8.906355703884504, -232263.61151457933, -856855.9125591302, 8.890917845306198, 8.903313531803876, 8.920573247332962, -245596.2804071747, 8.893279811454148, 8.937392538100541, -943699.860781345, 8.895671552085366, -1437665.0825195757, -12605.921556980806, 8.895069330184844, 8.95843297900801, 8.971863427023985, 9.024047637654226, 8.969213706203938, 8.878911873457389, -836695.1567627166, 8.839406663796643, 8.893476570353453, 8.965542449231794, 8.925218339075338, 9.003850421592333, 8.943464707947703, 8.664175074860744, -852864.7039046142, 8.892640502951593, 8.886680785073642, 8.971224025282936, 8.848377618425442, 8.883686962656808, 9.003337836244482, 8.868480118211025, -1295833.1161468895, -858229.695459076, -866626.5626293914, 8.834916255102824, 8.974353656015916, 8.856653704940637, 8.866045518481993, 8.869111953189346, 8.86956361749202, -2153130.262681779, 9.019376472579465, -858137.6643081908, -861899.4985045161, 8.909867000288797, -841825.6067221923, 8.906584025906467, -871426.659246311, 8.969714522936432, 8.962741917641578, 8.973909056384187, 8.928366456953041, 8.849943729812574, 8.931612643224703, 8.850270039660524, 8.844461127155517, 8.880999564151105, 8.903701840073238, 8.98390527340591, 8.896329871673812, 8.974647757189823, 8.89277327190362, 8.959985996033007, -185411.1975767088, 8.973341161387209, 8.806744755969026, 8.806918448830514, 8.877058444574251, -907274.8600337926, -2258893.7526855078, 8.938762357511141, 8.937213091509278]
'''
[11.903528092486384, -676620.4903798677, -608607.4951599997, 12.029834480718755, 11.95344655180705, 11.83608754553003, 12.066245638120229, -595005.0402565764, 12.115491795538643, 12.083211229361025, 12.132804416542886, -86893.04254448705, 11.992217217217362, -6090.541223037138, 12.01576586861482, -646302.3856839582, 12.012887235247222, -619674.2196366803, -594429.8757950818, -15415.19672711436, -79983.164028895, -44969.60752639504, 12.137563619562826, 12.0581715556512, 12.05409214416421, -476117.7576699087, -620904.448325301, 12.085068880187896, -82853.37812378624, 12.112355341643983, -457552.63074062846, 11.908303855556973, 12.058820867521142, -683834.9483319648, 11.951652594452938, -1041780.6817713513, -9129.168755757739, 12.09362103876871, 12.155976514329796, 12.117436450964263, -234883.2844228675, 12.092624425953218, 11.953634931702002, -606295.2485417414, 11.959708973423187, -3878.3767323358607, 12.18319977641229, 11.982136943929028, 12.146797623974537, 12.150322757356745, 11.819885148107739, -618012.3272351631, 12.006236135199991, 11.980697296292279, 12.024912850399424, -61108.02156229197, 12.004238192354387, 12.18797365436515, 11.978297888644517, -939003.8703648155, -621899.9479583297, -627984.7374397214, 12.017830633623268, 12.125977899649191, 11.932056895333105, 11.98892382313185, 12.050577554689305, 12.02221529088133, -1560233.7289255685, 12.147576316930436, -621833.2633539747, -624559.2975170928, 12.087508487865383, -610012.9807600962, 12.020721057128506, -631462.9674547213, 12.10972931191519, -76513.13064648554, 12.127194208007854, 12.049281252030235, -3898.0217094673903, -79132.49560663333, 12.019930540014292, 11.980425400989681, 12.008334046599208, 12.050880426823166, 12.143367368000533, 12.065603964126023, 12.136052521315692, 12.055524349141885, 12.121755120583995, -134350.4208357343, 12.096994225584513, 11.880603914205082, -163734.63501900306, -6301.189767113579, -657439.9346294891, -1636873.9664481136, 12.094326683947736, 12.114283184531182]
'''
positive_vel = []
for vel in velocity:
    if vel > 0:
        positive_vel.append(vel)

print(np.mean(positive_vel))


