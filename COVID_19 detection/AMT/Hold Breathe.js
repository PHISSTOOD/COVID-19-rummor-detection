<!-- You must include this JavaScript file -->
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

    <!-- For the full list of available Crowd HTML Elements and their input/output documentation,
    please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

    <!-- You must include crowd-form so that your task submits answers to MTurk -->
    <crowd-form answer-format="flatten-objects">

    <!-- The crowd-classifier element will create a tool for the Worker to select the
correct answer to your question -->
<crowd-classifier
categories="['Supporting rumors', 'Exposing rumors', 'Neither of first two categories(but related to the topic)','NOT related to the topic']"
header="Select the category for the tweet from these four categories"
name="detect">

    <!-- The text you want classified will be substituted for the "text" variable when you
publish a batch with a CSV input file containing multiple text items  -->
<classification-target> ${text} </classification-target>

<!-- Use the short-instructions section for quick instructions that the Worker
       will see while working on the task. Including some basic examples of
       good and bad answers here can help get good results. You can include
       any HTML here. -->
<short-instructions>
<h3><p>Please make sure you have viewed and understood the insructions before accepting task.</p></h3>
<p>The topic of this tweet is <strong>"Holding your breath is NOT a ‘simple self-check’ for coronavirus"</strong></p>
</short-instructions>
<!-- Use the full-instructions section for more detailed instructions that the
      Worker can open while working on the task. Including more detailed
      instructions and additional examples of good and bad answers here can
      help get good results. You can include any HTML here. -->
<full-instructions header="rumor detection instruction">
    <p><h3>The WHO has stated that "Holding your breath is NOT a ‘simple self-check’ for coronavirus".</h3> So please choose the type according to the content of the tweets,
if its meaning is revealing or denying the rumor, please choose "Exposing rumors", if it is supporting rumor, please select "Supporting rumors",
if it is neither of first two categories but related to the topic, please select "Neither of first two categories(but related to the topic)", if this tweet is not realted to the topic
please select "Not related to the topic"</p>
<p><strong>Example: </strong>Holding breath for 10 seconds without coughing will not guarantee you that you are not having corona . Already many corona patients hold their breath more than 12 seconds to price this wrong. Stop discussing with WhatsApp doctor in your studio</p>
<p><strong>Intent: </strong>Exposing rumors</p>
<p><strong>Example: </strong>Daily test against coronavirus is holding your breath for 10 sec if you are able to do so without much stress or coughing, you are good. The virus attacks the respiratory system so it's a good daily self-check.</p>
<p><strong>Intent: </strong>Supporting rumor</p>
<p><strong>Example: </strong>Holding your breath for 10 seconds without coughing detects coronavirus Is it a myth or truth? </p>
<p><strong>Intent: </strong>Neither of first two categories(but related to the topic)</p>
<p><strong>Example: </strong>Don't anyone hold their breath waiting for @DominicRaab to answer a question. </p>
<p><strong>Intent: </strong>NOT related to the topic</p>
</full-instructions>

</crowd-classifier>
</crowd-form>