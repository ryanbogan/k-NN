/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.opensearch.knn.quantization.enums.SQTypes;
import org.opensearch.knn.quantization.models.quantizationParams.SQParams;
import org.opensearch.knn.quantization.models.quantizationState.OneBitScalarQuantizationState;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KNNQuantizationStateWriter {
    public static void write(SegmentWriteState segmentWriteState, List<NativeEnginesKNNVectorsWriter.FieldWriter<?>> fields) throws IOException {
        String quantizationStateFileName =
                IndexFileNames.segmentFileName(
                        segmentWriteState.segmentInfo.name, segmentWriteState.segmentSuffix, "qs");

        IndexOutput output = segmentWriteState.directory.createOutput(quantizationStateFileName, segmentWriteState.context);
        CodecUtil.writeIndexHeader(output,"QuantizationCodec", 0,
                segmentWriteState.segmentInfo.getId(), segmentWriteState.segmentSuffix);
        // Collect data and positions
        List<FieldQuantizationState> fieldQuantizationStates = new ArrayList<>();
        List<Long> positions = new ArrayList<>();

        for(NativeEnginesKNNVectorsWriter.FieldWriter<?> field : fields) {
            FieldInfo fieldInfo = field.getFieldInfo();
            String fieldName = fieldInfo.getName();
            SQParams params = new SQParams(SQTypes.ONE_BIT);
            float[] thresholds = new float[]{
                    0.654869794845581F, 0.9052262902259827F, 0.9950445294380188F, 0.2792104482650757F, 0.9008876085281372F,
                    0.5784471035003662F, 0.9937006831169128F, 0.40283381938934326F, 0.5326775908470154F, 0.3075125515460968F,
                    0.21290498971939087F, 0.8878312110900879F, 0.7636076807975769F, 0.01169976219534874F, 0.3316563665866852F,
                    0.5756045579910278F, 0.18071097135543823F, 0.3180718719959259F, 0.6339374780654907F, 0.5606051683425903F,
                    0.8277460932731628F, 0.7976890802383423F, 0.06487838923931122F, 0.6465876698493958F, 0.22776716947555542F,
                    0.20427027344703674F, 0.9546729922294617F, 0.06582561135292053F, 0.671623706817627F, 0.7405648231506348F,
                    0.7188655138015747F, 0.522433876991272F, 0.21721620857715607F, 0.7059515118598938F, 0.7909319400787354F,
                    0.04052409902215004F, 0.48964083194732666F, 0.6994709372520447F, 0.025241823866963387F, 0.890583872795105F,
                    0.8092108964920044F, 0.5266998410224915F, 0.01794269122183323F, 0.5184373259544373F, 0.03063591942191124F,
                    0.8913800716400146F, 0.9000813364982605F, 0.7815365791320801F, 0.6454189419746399F, 0.748637318611145F,
                    0.3280772566795349F, 0.6496771574020386F, 0.2992631196975708F, 0.3802172541618347F, 0.45220163464546204F,
                    0.5711705684661865F, 0.5931774377822876F, 0.3590834140777588F, 0.06981778889894485F, 0.10721352696418762F,
                    0.8108277916908264F, 0.4067000150680542F, 0.48208147287368774F, 0.5513433218002319F, 0.9370184540748596F,
                    0.20080901682376862F, 0.2877649664878845F, 0.2174653857946396F, 0.1044510006904602F, 0.05650295689702034F,
                    0.1790303885936737F, 0.11731380969285965F, 0.7782679200172424F, 0.7417389154434204F, 0.01765206642448902F,
                    0.3107912838459015F, 0.8106122612953186F, 0.25063562393188477F, 0.9220748543739319F, 0.650646448135376F,
                    0.08959970623254776F, 0.02041495218873024F, 0.8796795010566711F, 0.49255120754241943F, 0.7198466668128967F,
                    0.5023201107978821F, 0.42582061886787415F, 0.3336160480976105F, 0.3254185914993286F, 0.9410670399665833F,
                    0.06591458618640804F, 0.80290687084198F, 0.043354880064725876F, 0.10248962885141373F, 0.7499788999557495F,
                    0.4327959716320038F, 0.3900298476219177F, 0.30700913071632385F, 0.5108733773231506F, 0.08815769106149673F,
                    0.8822945356369019F, 0.7951259613037109F, 0.709924042224884F, 0.8037276268005371F, 0.3892078101634979F,
                    0.719262421131134F, 0.937879741191864F, 0.6443655490875244F, 0.4834284484386444F, 0.6908601522445679F,
                    0.522384762763977F, 0.6067983508110046F, 0.1576155722141266F, 0.23843857645988464F, 0.23241622745990753F,
                    0.7124131917953491F, 0.22031652927303314F, 0.6513280272483826F, 0.4416605234146118F, 0.3915868401527405F,
                    0.6043621897697449F, 0.24115727841854095F, 0.47831958532333374F, 0.5678921937942505F, 0.29232627153396606F,
                    0.6508975028991699F, 0.2776077091693878F, 0.3272881507873535F, 0.7283280491828918F, 0.9820341467857361F,
                    0.6619274616241455F, 0.693916916847229F, 0.9192155590057373F, 0.6116933822631836F, 0.5121749639511108F,
                    0.9427587985992432F, 0.15219850838184357F, 0.8564485311508179F, 0.3897084593772888F, 0.3228261470794678F,
                    0.5957604646682739F, 0.692929744720459F, 0.148649662733078F, 0.8486062288284302F, 0.6429468989372253F,
                    0.1857590675354004F, 0.7208591690063477F, 0.7997427582740784F, 0.8402364253997803F, 0.6492679119110107F,
                    0.2197604775428772F, 0.9801485538482666F, 0.8821543455123901F, 0.991065263748169F, 0.4941260814666748F,
                    0.38822492957115173F, 0.34657055139541626F, 0.3791753947734833F, 0.6794843077659607F, 0.9947749376296997F,
                    0.17450153827667236F, 0.5909566283226013F, 0.7891755700111389F, 0.08722375333309174F, 0.6376811861991882F,
                    0.06368065625429153F, 0.21150462329387665F, 0.30261221528053284F, 0.6282126307487488F, 0.3822457492351532F,
                    0.6843811860084534F, 0.4975687563419342F, 0.37462571263313293F, 0.09784050285816193F, 0.8976691365242004F,
                    0.13591203093528748F, 0.42414507269859314F, 0.8468313217163086F, 0.2647649347782135F, 0.8022087216377258F,
                    0.24525234103298187F, 0.7648033490180969F, 0.13865691423416138F, 0.5153839588165283F, 0.9720014333724976F,
                    0.963642418384552F, 0.7043229341506958F, 0.5274141430854797F, 0.8185873627662659F, 0.018629500597715378F,
                    0.9978043437004089F, 0.49241843819618225F, 0.8358824253082275F, 0.15832984447479248F, 0.6892860531806946F,
                    0.3578386902809143F, 0.2228265404701233F, 0.33551591634750366F, 0.7593529224395752F, 0.8898977041244507F,
                    0.8015614748001099F, 0.9601506590843201F, 0.43443119525909424F, 0.9442645311355591F, 0.16680419445037842F,
                    0.9471819400787354F, 0.8222575187683105F, 0.3925081789493561F, 0.6578899621963501F, 0.5666460990905762F,
                    0.24499514603614807F, 0.47895172238349915F, 0.30516836047172546F, 0.013372616753935814F, 0.16724953055381775F,
                    0.23043709993362427F, 0.8988081216812134F, 0.981299102306366F, 0.20614632964134216F, 0.3586258590221405F,
                    0.3420194089412689F, 0.9417310953140259F, 0.6377236247062683F, 0.6942535638809204F, 0.6827767491340637F,
                    0.5289815664291382F, 0.26414138078689575F, 0.1976977288722992F, 0.1986723691225052F, 0.5644023418426514F,
                    0.20553550124168396F, 0.2564579248428345F, 0.06300272792577744F, 0.20989109575748444F, 0.44515904784202576F,
                    0.8690153956413269F, 0.2217683494091034F, 0.7604702711105347F, 0.4240388879776001F, 0.6497191786766052F,
                    0.21317964792251587F, 0.28608107566833496F, 0.9205999979972839F, 0.6816153526306152F, 0.431233286857605F,
                    0.7636179327964783F, 0.7488064160346985F, 0.271390438079834F, 0.34144195914268494F, 0.9407476186752319F,
                    0.41091084480285645F, 0.2380474954843521F, 0.8268464803695679F, 0.20072944462299347F, 0.6159149408340454F,
                    0.8982447385787964F, 0.0634542927145958F, 0.2623581290245056F, 0.7974768877029419F, 0.22598734402656555F,
                    0.6233339905738831F, 0.4512479305267334F, 0.16651110315322876F, 0.8166211247444153F, 0.5519981980323792F,
                    0.20018723607063293F, 0.6335357427597046F, 0.6116283535957336F, 0.7946094274520874F, 0.47493410110473633F,
                    0.3729104995727539F, 0.7878749370574951F, 0.383637011051178F, 0.3701019883155823F, 0.03586839884519577F,
                    0.9981721639633179F, 0.24044327437877655F, 0.0617859922349453F, 0.14014728367328644F, 0.7614969611167908F,
                    0.15536275506019592F, 0.39315956830978394F, 0.017467450350522995F, 0.18591895699501038F, 0.820717990398407F,
                    0.5762999057769775F, 0.38212376832962036F, 0.7556784749031067F, 0.9927763938903809F, 0.237848162651062F,
                    0.5688667893409729F, 0.7840257287025452F, 0.9180153012275696F, 0.7492987513542175F, 0.8117421865463257F,
                    0.4331830143928528F, 0.6535963416099548F, 0.24942384660243988F, 0.9825491309165955F, 0.4123951494693756F,
                    0.7157326936721802F, 0.4592820107936859F, 0.4181206524372101F, 0.46307137608528137F, 0.5307919383049011F,
                    0.23410895466804504F, 0.9564872980117798F, 0.5173934698104858F, 0.4334239959716797F, 0.03643488883972168F,
                    0.06798890233039856F, 0.20293556153774261F, 0.3132593035697937F, 0.43198782205581665F, 0.39346256852149963F,
                    0.726813554763794F, 0.9737647771835327F, 0.5204455852508545F, 0.3138745427131653F, 0.6600716710090637F,
                    0.9642258286476135F, 0.9349855780601501F, 0.8255032896995544F, 0.2728443145751953F, 0.8708490133285522F,
                    0.8473383784294128F, 0.4429991841316223F, 0.47430452704429626F, 0.663441002368927F, 0.08088516443967819F,
                    0.7478584051132202F, 0.5714508295059204F, 0.19590604305267334F, 0.23807664215564728F, 0.40367233753204346F,
                    0.47966745495796204F, 0.3857768177986145F, 0.770660936832428F, 0.8154998421669006F, 0.8362643122673035F,
                    0.3736884593963623F, 0.845207154750824F, 0.6117952466011047F, 0.17094875872135162F, 0.10966288298368454F,
                    0.654869794845581F, 0.9052262902259827F, 0.9950445294380188F, 0.2792104482650757F, 0.9008876085281372F,
                    0.5784471035003662F, 0.9937006831169128F, 0.40283381938934326F, 0.5326775908470154F, 0.3075125515460968F,
                    0.21290498971939087F, 0.8878312110900879F, 0.7636076807975769F, 0.01169976219534874F, 0.3316563665866852F,
                    0.5756045579910278F, 0.18071097135543823F, 0.3180718719959259F, 0.6339374780654907F, 0.5606051683425903F,
                    0.8277460932731628F, 0.7976890802383423F, 0.06487838923931122F, 0.6465876698493958F, 0.22776716947555542F,
                    0.20427027344703674F, 0.9546729922294617F, 0.06582561135292053F, 0.671623706817627F, 0.7405648231506348F,
                    0.7188655138015747F, 0.522433876991272F, 0.21721620857715607F, 0.7059515118598938F, 0.7909319400787354F,
                    0.04052409902215004F, 0.48964083194732666F, 0.6994709372520447F, 0.025241823866963387F, 0.890583872795105F,
                    0.8092108964920044F, 0.5266998410224915F, 0.01794269122183323F, 0.5184373259544373F, 0.03063591942191124F,
                    0.8913800716400146F, 0.9000813364982605F, 0.7815365791320801F, 0.6454189419746399F, 0.748637318611145F,
                    0.3280772566795349F, 0.6496771574020386F, 0.2992631196975708F, 0.3802172541618347F, 0.45220163464546204F,
                    0.5711705684661865F, 0.5931774377822876F, 0.3590834140777588F, 0.06981778889894485F, 0.10721352696418762F,
                    0.8108277916908264F, 0.4067000150680542F, 0.48208147287368774F, 0.5513433218002319F, 0.9370184540748596F,
                    0.20080901682376862F, 0.2877649664878845F, 0.2174653857946396F, 0.1044510006904602F, 0.05650295689702034F,
                    0.1790303885936737F, 0.11731380969285965F, 0.7782679200172424F, 0.7417389154434204F, 0.01765206642448902F,
                    0.3107912838459015F, 0.8106122612953186F, 0.25063562393188477F, 0.9220748543739319F, 0.650646448135376F,
                    0.08959970623254776F, 0.02041495218873024F, 0.8796795010566711F, 0.49255120754241943F, 0.7198466668128967F,
                    0.5023201107978821F, 0.42582061886787415F, 0.3336160480976105F, 0.3254185914993286F, 0.9410670399665833F,
                    0.06591458618640804F, 0.80290687084198F, 0.043354880064725876F, 0.10248962885141373F, 0.7499788999557495F,
                    0.4327959716320038F, 0.3900298476219177F, 0.30700913071632385F, 0.5108733773231506F, 0.08815769106149673F,
                    0.8822945356369019F, 0.7951259613037109F, 0.709924042224884F, 0.8037276268005371F, 0.3892078101634979F,
                    0.719262421131134F, 0.937879741191864F, 0.6443655490875244F, 0.4834284484386444F, 0.6908601522445679F,
                    0.522384762763977F, 0.6067983508110046F, 0.1576155722141266F, 0.23843857645988464F, 0.23241622745990753F,
                    0.7124131917953491F, 0.22031652927303314F, 0.6513280272483826F, 0.4416605234146118F, 0.3915868401527405F,
                    0.6043621897697449F, 0.24115727841854095F, 0.47831958532333374F, 0.5678921937942505F, 0.29232627153396606F,
                    0.6508975028991699F, 0.2776077091693878F, 0.3272881507873535F, 0.7283280491828918F, 0.9820341467857361F,
                    0.6619274616241455F, 0.693916916847229F, 0.9192155590057373F, 0.6116933822631836F, 0.5121749639511108F,
                    0.9427587985992432F, 0.15219850838184357F, 0.8564485311508179F, 0.3897084593772888F, 0.3228261470794678F,
                    0.5957604646682739F, 0.692929744720459F, 0.148649662733078F, 0.8486062288284302F, 0.6429468989372253F,
                    0.1857590675354004F, 0.7208591690063477F, 0.7997427582740784F, 0.8402364253997803F, 0.6492679119110107F,
                    0.2197604775428772F, 0.9801485538482666F, 0.8821543455123901F, 0.991065263748169F, 0.4941260814666748F,
                    0.38822492957115173F, 0.34657055139541626F, 0.3791753947734833F, 0.6794843077659607F, 0.9947749376296997F,
                    0.17450153827667236F, 0.5909566283226013F, 0.7891755700111389F, 0.08722375333309174F, 0.6376811861991882F,
                    0.06368065625429153F, 0.21150462329387665F, 0.30261221528053284F, 0.6282126307487488F, 0.3822457492351532F,
                    0.6843811860084534F, 0.4975687563419342F, 0.37462571263313293F, 0.09784050285816193F, 0.8976691365242004F,
                    0.13591203093528748F, 0.42414507269859314F, 0.8468313217163086F, 0.2647649347782135F, 0.8022087216377258F,
                    0.24525234103298187F, 0.7648033490180969F, 0.13865691423416138F, 0.5153839588165283F, 0.9720014333724976F,
                    0.963642418384552F, 0.7043229341506958F, 0.5274141430854797F, 0.8185873627662659F, 0.018629500597715378F,
                    0.9978043437004089F, 0.49241843819618225F, 0.8358824253082275F, 0.15832984447479248F, 0.6892860531806946F,
                    0.3578386902809143F, 0.2228265404701233F, 0.33551591634750366F, 0.7593529224395752F, 0.8898977041244507F,
                    0.8015614748001099F, 0.9601506590843201F, 0.43443119525909424F, 0.9442645311355591F, 0.16680419445037842F,
                    0.9471819400787354F, 0.8222575187683105F, 0.3925081789493561F, 0.6578899621963501F, 0.5666460990905762F,
                    0.24499514603614807F, 0.47895172238349915F, 0.30516836047172546F, 0.013372616753935814F, 0.16724953055381775F,
                    0.23043709993362427F, 0.8988081216812134F, 0.981299102306366F, 0.20614632964134216F, 0.3586258590221405F,
                    0.3420194089412689F, 0.9417310953140259F, 0.6377236247062683F, 0.6942535638809204F, 0.6827767491340637F,
                    0.5289815664291382F, 0.26414138078689575F, 0.1976977288722992F, 0.1986723691225052F, 0.5644023418426514F,
                    0.20553550124168396F, 0.2564579248428345F, 0.06300272792577744F, 0.20989109575748444F, 0.44515904784202576F,
                    0.8690153956413269F, 0.2217683494091034F, 0.7604702711105347F, 0.4240388879776001F, 0.6497191786766052F,
                    0.21317964792251587F, 0.28608107566833496F, 0.9205999979972839F, 0.6816153526306152F, 0.431233286857605F,
                    0.7636179327964783F, 0.7488064160346985F, 0.271390438079834F, 0.34144195914268494F, 0.9407476186752319F,
                    0.41091084480285645F, 0.2380474954843521F, 0.8268464803695679F, 0.20072944462299347F, 0.6159149408340454F,
                    0.8982447385787964F, 0.0634542927145958F, 0.2623581290245056F, 0.7974768877029419F, 0.22598734402656555F,
                    0.6233339905738831F, 0.4512479305267334F, 0.16651110315322876F, 0.8166211247444153F, 0.5519981980323792F,
                    0.20018723607063293F, 0.6335357427597046F, 0.6116283535957336F, 0.7946094274520874F, 0.47493410110473633F,
                    0.3729104995727539F, 0.7878749370574951F, 0.383637011051178F, 0.3701019883155823F, 0.03586839884519577F,
                    0.9981721639633179F, 0.24044327437877655F, 0.0617859922349453F, 0.14014728367328644F, 0.7614969611167908F,
                    0.15536275506019592F, 0.39315956830978394F, 0.017467450350522995F, 0.18591895699501038F, 0.820717990398407F,
                    0.5762999057769775F, 0.38212376832962036F, 0.7556784749031067F, 0.9927763938903809F, 0.237848162651062F,
                    0.5688667893409729F, 0.7840257287025452F, 0.9180153012275696F, 0.7492987513542175F, 0.8117421865463257F,
                    0.4331830143928528F, 0.6535963416099548F, 0.24942384660243988F, 0.9825491309165955F, 0.4123951494693756F,
                    0.7157326936721802F, 0.4592820107936859F, 0.4181206524372101F, 0.46307137608528137F, 0.5307919383049011F,
                    0.23410895466804504F, 0.9564872980117798F, 0.5173934698104858F, 0.4334239959716797F, 0.03643488883972168F,
                    0.06798890233039856F, 0.20293556153774261F, 0.3132593035697937F, 0.43198782205581665F, 0.39346256852149963F,
                    0.726813554763794F, 0.9737647771835327F, 0.5204455852508545F, 0.3138745427131653F, 0.6600716710090637F,
                    0.9642258286476135F, 0.9349855780601501F, 0.8255032896995544F, 0.2728443145751953F, 0.8708490133285522F,
                    0.8473383784294128F, 0.4429991841316223F, 0.47430452704429626F, 0.663441002368927F, 0.08088516443967819F,
                    0.7478584051132202F, 0.5714508295059204F, 0.19590604305267334F, 0.23807664215564728F, 0.40367233753204346F,
                    0.47966745495796204F, 0.3857768177986145F, 0.770660936832428F, 0.8154998421669006F, 0.8362643122673035F,
                    0.3736884593963623F, 0.845207154750824F, 0.6117952466011047F, 0.17094875872135162F, 0.10966288298368454F,
                    0.654869794845581F, 0.9052262902259827F, 0.9950445294380188F, 0.2792104482650757F, 0.9008876085281372F,
                    0.5784471035003662F, 0.9937006831169128F, 0.40283381938934326F, 0.5326775908470154F, 0.3075125515460968F,
                    0.21290498971939087F, 0.8878312110900879F, 0.7636076807975769F, 0.01169976219534874F, 0.3316563665866852F,
                    0.5756045579910278F, 0.18071097135543823F, 0.3180718719959259F, 0.6339374780654907F, 0.5606051683425903F,
                    0.8277460932731628F, 0.7976890802383423F, 0.06487838923931122F, 0.6465876698493958F, 0.22776716947555542F,
                    0.20427027344703674F, 0.9546729922294617F, 0.06582561135292053F, 0.671623706817627F, 0.7405648231506348F,
                    0.7188655138015747F, 0.522433876991272F, 0.21721620857715607F, 0.7059515118598938F, 0.7909319400787354F,
                    0.04052409902215004F, 0.48964083194732666F, 0.6994709372520447F, 0.025241823866963387F, 0.890583872795105F,
                    0.8092108964920044F, 0.5266998410224915F, 0.01794269122183323F, 0.5184373259544373F, 0.03063591942191124F,
                    0.8913800716400146F, 0.9000813364982605F, 0.7815365791320801F, 0.6454189419746399F, 0.748637318611145F,
                    0.3280772566795349F, 0.6496771574020386F, 0.2992631196975708F, 0.3802172541618347F, 0.45220163464546204F,
                    0.5711705684661865F, 0.5931774377822876F, 0.3590834140777588F, 0.06981778889894485F, 0.10721352696418762F,
                    0.8108277916908264F, 0.4067000150680542F, 0.48208147287368774F, 0.5513433218002319F, 0.9370184540748596F,
                    0.20080901682376862F, 0.2877649664878845F, 0.2174653857946396F, 0.1044510006904602F, 0.05650295689702034F,
                    0.1790303885936737F, 0.11731380969285965F, 0.7782679200172424F, 0.7417389154434204F, 0.01765206642448902F,
                    0.3107912838459015F, 0.8106122612953186F, 0.25063562393188477F, 0.9220748543739319F, 0.650646448135376F,
                    0.08959970623254776F, 0.02041495218873024F, 0.8796795010566711F, 0.49255120754241943F, 0.7198466668128967F,
                    0.5023201107978821F, 0.42582061886787415F, 0.3336160480976105F, 0.3254185914993286F, 0.9410670399665833F,
                    0.06591458618640804F, 0.80290687084198F, 0.043354880064725876F, 0.10248962885141373F, 0.7499788999557495F,
                    0.4327959716320038F, 0.3900298476219177F, 0.30700913071632385F, 0.5108733773231506F, 0.08815769106149673F,
                    0.8822945356369019F, 0.7951259613037109F, 0.709924042224884F, 0.8037276268005371F, 0.3892078101634979F,
                    0.719262421131134F, 0.937879741191864F, 0.6443655490875244F, 0.4834284484386444F, 0.6908601522445679F,
                    0.522384762763977F, 0.6067983508110046F, 0.1576155722141266F, 0.23843857645988464F, 0.23241622745990753F,
                    0.7124131917953491F, 0.22031652927303314F, 0.6513280272483826F, 0.4416605234146118F, 0.3915868401527405F,
                    0.6043621897697449F, 0.24115727841854095F, 0.47831958532333374F, 0.5678921937942505F, 0.29232627153396606F,
                    0.6508975028991699F, 0.2776077091693878F, 0.3272881507873535F, 0.7283280491828918F, 0.9820341467857361F,
                    0.6619274616241455F, 0.693916916847229F, 0.9192155590057373F, 0.6116933822631836F, 0.5121749639511108F,
                    0.9427587985992432F, 0.15219850838184357F, 0.8564485311508179F, 0.3897084593772888F, 0.3228261470794678F,
                    0.5957604646682739F, 0.692929744720459F, 0.148649662733078F, 0.8486062288284302F, 0.6429468989372253F,
                    0.1857590675354004F, 0.7208591690063477F, 0.7997427582740784F, 0.8402364253997803F, 0.6492679119110107F,
                    0.2197604775428772F, 0.9801485538482666F, 0.8821543455123901F, 0.991065263748169F, 0.4941260814666748F,
                    0.38822492957115173F, 0.34657055139541626F, 0.3791753947734833F, 0.6794843077659607F, 0.9947749376296997F,
                    0.17450153827667236F, 0.5909566283226013F, 0.7891755700111389F, 0.08722375333309174F, 0.6376811861991882F,
                    0.06368065625429153F, 0.21150462329387665F, 0.30261221528053284F, 0.6282126307487488F, 0.3822457492351532F,
                    0.6843811860084534F, 0.4975687563419342F, 0.37462571263313293F, 0.09784050285816193F, 0.8976691365242004F,
                    0.13591203093528748F, 0.42414507269859314F, 0.8468313217163086F, 0.2647649347782135F, 0.8022087216377258F,
                    0.24525234103298187F, 0.7648033490180969F, 0.13865691423416138F, 0.5153839588165283F, 0.9720014333724976F,
                    0.963642418384552F, 0.7043229341506958F, 0.5274141430854797F, 0.8185873627662659F, 0.018629500597715378F,
                    0.9978043437004089F, 0.49241843819618225F, 0.8358824253082275F, 0.15832984447479248F, 0.6892860531806946F,
                    0.3578386902809143F, 0.2228265404701233F, 0.33551591634750366F, 0.7593529224395752F, 0.8898977041244507F,
                    0.8015614748001099F, 0.9601506590843201F, 0.43443119525909424F, 0.9442645311355591F, 0.16680419445037842F,
                    0.9471819400787354F, 0.8222575187683105F, 0.3925081789493561F, 0.6578899621963501F, 0.5666460990905762F,
                    0.24499514603614807F, 0.47895172238349915F, 0.30516836047172546F, 0.013372616753935814F, 0.16724953055381775F,
                    0.23043709993362427F, 0.8988081216812134F, 0.981299102306366F, 0.20614632964134216F, 0.3586258590221405F,
                    0.3420194089412689F, 0.9417310953140259F, 0.6377236247062683F, 0.6942535638809204F, 0.6827767491340637F,
                    0.5289815664291382F, 0.26414138078689575F, 0.1976977288722992F, 0.1986723691225052F, 0.5644023418426514F,
                    0.20553550124168396F, 0.2564579248428345F, 0.06300272792577744F, 0.20989109575748444F, 0.44515904784202576F,
                    0.8690153956413269F, 0.2217683494091034F, 0.7604702711105347F, 0.4240388879776001F, 0.6497191786766052F,
                    0.21317964792251587F, 0.28608107566833496F, 0.9205999979972839F, 0.6816153526306152F, 0.431233286857605F,
                    0.7636179327964783F, 0.7488064160346985F, 0.271390438079834F, 0.34144195914268494F, 0.9407476186752319F,
                    0.41091084480285645F, 0.2380474954843521F, 0.8268464803695679F, 0.20072944462299347F, 0.6159149408340454F,
                    0.8982447385787964F, 0.0634542927145958F, 0.2623581290245056F, 0.7974768877029419F, 0.22598734402656555F,
                    0.6233339905738831F, 0.4512479305267334F, 0.16651110315322876F, 0.8166211247444153F, 0.5519981980323792F,
                    0.20018723607063293F, 0.6335357427597046F, 0.6116283535957336F, 0.7946094274520874F, 0.47493410110473633F,
                    0.3729104995727539F, 0.7878749370574951F, 0.383637011051178F, 0.3701019883155823F, 0.03586839884519577F,
                    0.9981721639633179F, 0.24044327437877655F, 0.0617859922349453F, 0.14014728367328644F, 0.7614969611167908F,
                    0.15536275506019592F, 0.39315956830978394F, 0.017467450350522995F, 0.18591895699501038F, 0.820717990398407F,
                    0.5762999057769775F, 0.38212376832962036F, 0.7556784749031067F, 0.9927763938903809F, 0.237848162651062F,
                    0.5688667893409729F, 0.7840257287025452F, 0.9180153012275696F, 0.7492987513542175F, 0.8117421865463257F,
                    0.4331830143928528F, 0.6535963416099548F, 0.24942384660243988F, 0.9825491309165955F, 0.4123951494693756F,
                    0.7157326936721802F, 0.4592820107936859F, 0.4181206524372101F, 0.46307137608528137F, 0.5307919383049011F,
                    0.23410895466804504F, 0.9564872980117798F, 0.5173934698104858F, 0.4334239959716797F, 0.03643488883972168F,
                    0.06798890233039856F, 0.20293556153774261F, 0.3132593035697937F, 0.43198782205581665F, 0.39346256852149963F,
                    0.726813554763794F, 0.9737647771835327F, 0.5204455852508545F, 0.3138745427131653F, 0.6600716710090637F,
                    0.9642258286476135F, 0.9349855780601501F, 0.8255032896995544F, 0.2728443145751953F, 0.8708490133285522F,
                    0.8473383784294128F, 0.4429991841316223F, 0.47430452704429626F, 0.663441002368927F, 0.08088516443967819F,
                    0.7478584051132202F, 0.5714508295059204F, 0.19590604305267334F, 0.23807664215564728F, 0.40367233753204346F,
                    0.47966745495796204F, 0.3857768177986145F, 0.770660936832428F, 0.8154998421669006F, 0.8362643122673035F,
                    0.3736884593963623F, 0.845207154750824F, 0.6117952466011047F, 0.17094875872135162F, 0.10966288298368454F,
                    0.654869794845581F, 0.9052262902259827F, 0.9950445294380188F, 0.2792104482650757F, 0.9008876085281372F,
                    0.5784471035003662F, 0.9937006831169128F, 0.40283381938934326F, 0.5326775908470154F, 0.3075125515460968F,
                    0.21290498971939087F, 0.8878312110900879F, 0.7636076807975769F, 0.01169976219534874F, 0.3316563665866852F,
                    0.5756045579910278F, 0.18071097135543823F, 0.3180718719959259F, 0.6339374780654907F, 0.5606051683425903F,
                    0.8277460932731628F, 0.7976890802383423F, 0.06487838923931122F, 0.6465876698493958F, 0.22776716947555542F,
                    0.20427027344703674F, 0.9546729922294617F, 0.06582561135292053F, 0.671623706817627F, 0.7405648231506348F,
                    0.7188655138015747F, 0.522433876991272F, 0.21721620857715607F, 0.7059515118598938F, 0.7909319400787354F,
                    0.04052409902215004F, 0.48964083194732666F, 0.6994709372520447F, 0.025241823866963387F, 0.890583872795105F,
                    0.8092108964920044F, 0.5266998410224915F, 0.01794269122183323F, 0.5184373259544373F, 0.03063591942191124F,
                    0.8913800716400146F, 0.9000813364982605F, 0.7815365791320801F, 0.6454189419746399F, 0.748637318611145F,
                    0.3280772566795349F, 0.6496771574020386F, 0.2992631196975708F, 0.3802172541618347F, 0.45220163464546204F,
                    0.5711705684661865F, 0.5931774377822876F, 0.3590834140777588F, 0.06981778889894485F, 0.10721352696418762F,
                    0.8108277916908264F, 0.4067000150680542F, 0.48208147287368774F, 0.5513433218002319F, 0.9370184540748596F,
                    0.20080901682376862F, 0.2877649664878845F, 0.2174653857946396F, 0.1044510006904602F, 0.05650295689702034F,
                    0.1790303885936737F, 0.11731380969285965F, 0.7782679200172424F, 0.7417389154434204F, 0.01765206642448902F,
                    0.3107912838459015F, 0.8106122612953186F, 0.25063562393188477F, 0.9220748543739319F, 0.650646448135376F,
                    0.08959970623254776F, 0.02041495218873024F, 0.8796795010566711F, 0.49255120754241943F, 0.7198466668128967F,
                    0.5023201107978821F, 0.42582061886787415F, 0.3336160480976105F, 0.3254185914993286F, 0.9410670399665833F,
                    0.06591458618640804F, 0.80290687084198F, 0.043354880064725876F, 0.10248962885141373F, 0.7499788999557495F,
                    0.4327959716320038F, 0.3900298476219177F, 0.30700913071632385F, 0.5108733773231506F, 0.08815769106149673F,
                    0.8822945356369019F, 0.7951259613037109F, 0.709924042224884F, 0.8037276268005371F, 0.3892078101634979F,
                    0.719262421131134F, 0.937879741191864F, 0.6443655490875244F, 0.4834284484386444F, 0.6908601522445679F,
                    0.522384762763977F, 0.6067983508110046F, 0.1576155722141266F, 0.23843857645988464F, 0.23241622745990753F,
                    0.7124131917953491F, 0.22031652927303314F, 0.6513280272483826F, 0.4416605234146118F, 0.3915868401527405F,
                    0.6043621897697449F, 0.24115727841854095F, 0.47831958532333374F, 0.5678921937942505F, 0.29232627153396606F,
                    0.6508975028991699F, 0.2776077091693878F, 0.3272881507873535F, 0.7283280491828918F, 0.9820341467857361F,
                    0.6619274616241455F, 0.693916916847229F, 0.9192155590057373F, 0.6116933822631836F, 0.5121749639511108F,
                    0.9427587985992432F, 0.15219850838184357F, 0.8564485311508179F, 0.3897084593772888F, 0.3228261470794678F,
                    0.5957604646682739F, 0.692929744720459F, 0.148649662733078F, 0.8486062288284302F, 0.6429468989372253F,
                    0.1857590675354004F, 0.7208591690063477F, 0.7997427582740784F, 0.8402364253997803F, 0.6492679119110107F,
                    0.2197604775428772F, 0.9801485538482666F, 0.8821543455123901F, 0.991065263748169F, 0.4941260814666748F,
                    0.38822492957115173F, 0.34657055139541626F, 0.3791753947734833F, 0.6794843077659607F, 0.9947749376296997F,
                    0.17450153827667236F, 0.5909566283226013F, 0.7891755700111389F, 0.08722375333309174F, 0.6376811861991882F,
                    0.06368065625429153F, 0.21150462329387665F, 0.30261221528053284F, 0.6282126307487488F, 0.3822457492351532F,
                    0.6843811860084534F, 0.4975687563419342F, 0.37462571263313293F, 0.09784050285816193F, 0.8976691365242004F,
                    0.13591203093528748F, 0.42414507269859314F, 0.8468313217163086F, 0.2647649347782135F, 0.8022087216377258F,
                    0.24525234103298187F, 0.7648033490180969F, 0.13865691423416138F, 0.5153839588165283F, 0.9720014333724976F,
                    0.963642418384552F, 0.7043229341506958F, 0.5274141430854797F, 0.8185873627662659F, 0.018629500597715378F,
                    0.9978043437004089F, 0.49241843819618225F, 0.8358824253082275F, 0.15832984447479248F, 0.6892860531806946F,
                    0.3578386902809143F, 0.2228265404701233F, 0.33551591634750366F, 0.7593529224395752F, 0.8898977041244507F,
                    0.8015614748001099F, 0.9601506590843201F, 0.43443119525909424F, 0.9442645311355591F, 0.16680419445037842F,
                    0.9471819400787354F, 0.8222575187683105F, 0.3925081789493561F, 0.6578899621963501F, 0.5666460990905762F,
                    0.24499514603614807F, 0.47895172238349915F, 0.30516836047172546F, 0.013372616753935814F, 0.16724953055381775F,
                    0.23043709993362427F, 0.8988081216812134F, 0.981299102306366F, 0.20614632964134216F, 0.3586258590221405F,
                    0.3420194089412689F, 0.9417310953140259F, 0.6377236247062683F, 0.6942535638809204F, 0.6827767491340637F,
                    0.5289815664291382F, 0.26414138078689575F, 0.1976977288722992F, 0.1986723691225052F, 0.5644023418426514F,
                    0.20553550124168396F, 0.2564579248428345F, 0.06300272792577744F, 0.20989109575748444F, 0.44515904784202576F,
                    0.8690153956413269F, 0.2217683494091034F, 0.7604702711105347F, 0.4240388879776001F, 0.6497191786766052F,
                    0.21317964792251587F, 0.28608107566833496F, 0.9205999979972839F, 0.6816153526306152F, 0.431233286857605F,
                    0.7636179327964783F, 0.7488064160346985F, 0.271390438079834F, 0.34144195914268494F, 0.9407476186752319F,
                    0.41091084480285645F, 0.2380474954843521F, 0.8268464803695679F, 0.20072944462299347F, 0.6159149408340454F,
                    0.8982447385787964F, 0.0634542927145958F, 0.2623581290245056F, 0.7974768877029419F, 0.22598734402656555F,
                    0.6233339905738831F, 0.4512479305267334F, 0.16651110315322876F, 0.8166211247444153F, 0.5519981980323792F,
                    0.20018723607063293F, 0.6335357427597046F, 0.6116283535957336F, 0.7946094274520874F, 0.47493410110473633F,
                    0.3729104995727539F, 0.7878749370574951F, 0.383637011051178F, 0.3701019883155823F, 0.03586839884519577F,
                    0.9981721639633179F, 0.24044327437877655F, 0.0617859922349453F, 0.14014728367328644F, 0.7614969611167908F,
                    0.15536275506019592F, 0.39315956830978394F, 0.017467450350522995F, 0.18591895699501038F, 0.820717990398407F,
                    0.5762999057769775F, 0.38212376832962036F, 0.7556784749031067F, 0.9927763938903809F, 0.237848162651062F,
                    0.5688667893409729F, 0.7840257287025452F, 0.9180153012275696F, 0.7492987513542175F, 0.8117421865463257F,
                    0.4331830143928528F, 0.6535963416099548F, 0.24942384660243988F, 0.9825491309165955F, 0.4123951494693756F,
                    0.7157326936721802F, 0.4592820107936859F, 0.4181206524372101F, 0.46307137608528137F, 0.5307919383049011F,
                    0.23410895466804504F, 0.9564872980117798F, 0.5173934698104858F, 0.4334239959716797F, 0.03643488883972168F,
                    0.06798890233039856F, 0.20293556153774261F, 0.3132593035697937F, 0.43198782205581665F, 0.39346256852149963F,
                    0.726813554763794F, 0.9737647771835327F, 0.5204455852508545F, 0.3138745427131653F, 0.6600716710090637F,
                    0.9642258286476135F, 0.9349855780601501F, 0.8255032896995544F, 0.2728443145751953F, 0.8708490133285522F,
                    0.8473383784294128F, 0.4429991841316223F, 0.47430452704429626F, 0.663441002368927F, 0.08088516443967819F,
                    0.7478584051132202F, 0.5714508295059204F, 0.19590604305267334F, 0.23807664215564728F, 0.40367233753204346F,
                    0.47966745495796204F, 0.3857768177986145F, 0.770660936832428F, 0.8154998421669006F, 0.8362643122673035F,
                    0.3736884593963623F, 0.845207154750824F, 0.6117952466011047F, 0.17094875872135162F, 0.10966288298368454F
            };
            OneBitScalarQuantizationState quantizationState = new OneBitScalarQuantizationState(params, thresholds);
            byte[] stateBytes = quantizationState.toByteArray();
            long position = output.getFilePointer();
            positions.add(position);

            output.writeBytes(stateBytes, stateBytes.length);
            fieldQuantizationStates.add(new FieldQuantizationState(fieldName, stateBytes));
        }

        // Now write the index section at the end
        long indexStartPosition = output.getFilePointer();
        output.writeInt(fieldQuantizationStates.size());
        for (int i1 = 0; i1 < fieldQuantizationStates.size(); i1++) {
            output.writeString(fieldQuantizationStates.get(i1).fieldName);
            output.writeInt(fieldQuantizationStates.get(i1).stateBytes.length);
            output.writeVLong(positions.get(i1));
        }
        output.writeLong(indexStartPosition);
        output.writeInt(-1);
        CodecUtil.writeFooter(output);
        output.close();
    }

    private static class FieldQuantizationState {
        String fieldName;
        byte[] stateBytes;

        FieldQuantizationState(String fieldName, byte[] stateBytes) {
            this.fieldName = fieldName;
            this.stateBytes = stateBytes;
        }
    }
}
